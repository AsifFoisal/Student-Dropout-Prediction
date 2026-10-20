"""Internal notebook generator (not part of the Kaggle project zip).

Rebuild the Kaggle notebook from the reusable project files:

    python kaggle/_build_notebook.py

The notebook embeds src/*.py and app/chatbot_app.py, so rerun this after editing
those files, README.md, or requirements.txt.
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent

FEATURE_ENGINEERING = (PROJECT / "src" / "feature_engineering.py").read_text(encoding="utf-8")
PREPROCESSING = (PROJECT / "src" / "preprocessing.py").read_text(encoding="utf-8")
PREDICT = (PROJECT / "src" / "predict.py").read_text(encoding="utf-8")
APP = (PROJECT / "app" / "chatbot_app.py").read_text(encoding="utf-8")
README = (PROJECT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = (PROJECT / "requirements.txt").read_text(encoding="utf-8")

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


md(
    """
# Student Dropout Prediction and Explainable Assistant

**Kaggle edition.** This notebook goes from the raw CSV to a saved, explainable
XGBoost model and an interactive risk assistant.

## How to run

1. Add `dataset_outliers_removed_balanced.csv` as a Kaggle dataset.
2. Open this notebook in Kaggle.
3. Click **Run all**. A CPU is enough; no GPU is required.
4. Use the widget form near the bottom to get a prediction and explanation.
5. Download `student_dropout_project.zip` from `/kaggle/working`.

## What you get

- A trained XGBoost dropout model
- A saved preprocessing object, feature list, threshold, category map, metadata
- Evaluation report, XGBoost importance, global SHAP, category contribution
- `src/` modules for later local use
- A Streamlit app for the local shift
- An ipywidgets chatbot that works inside Kaggle

All preprocessing is fitted on the training split only. The test split is used
only for final evaluation. `RANDOM_STATE = 42`.
"""
)

md(
    """
## Step 0 - Setup

This cell imports everything once, creates the project folders, finds the CSV,
and moves into the project directory. If the dataset is not found, add it as a
Kaggle dataset and rerun this cell.
"""
)

code(
    r'''
import os, sys, json, glob, shutil, zipfile, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
import ipywidgets as widgets
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
SOURCE_DIR = Path.cwd()
if Path("/kaggle/working").exists():
    PROJECT_DIR = Path("/kaggle/working/student_dropout_project")
else:
    PROJECT_DIR = SOURCE_DIR / "student_dropout_project"

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SRC_DIR = PROJECT_DIR / "src"
APP_DIR = PROJECT_DIR / "app"
for folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SRC_DIR, APP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
(SRC_DIR / "__init__.py").write_text(
    '"""Student dropout prediction package."""\n', encoding="utf-8"
)


def find_dataset():
    roots = [Path("/kaggle/input"), SOURCE_DIR, Path("/kaggle/working")]
    hits = []
    for root in roots:
        if root.exists():
            hits.extend(root.rglob("dataset_outliers_removed_balanced.csv"))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(
            "Add dataset_outliers_removed_balanced.csv as a Kaggle dataset, "
            "then rerun this cell."
        )
    return hits[0]


DATASET_PATH = find_dataset()
LOCAL_DATA_PATH = DATA_DIR / DATASET_PATH.name
shutil.copy2(DATASET_PATH, LOCAL_DATA_PATH)

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

RISK_BANDS = {
    "Low": [0.00, 0.35],
    "Medium": [0.35, 0.65],
    "High": [0.65, 0.85],
    "Critical": [0.85, 1.01],
}

print("Dataset:", DATASET_PATH)
print("Project:", PROJECT_DIR)
print("Python:", sys.version.split()[0])
print("xgboost:", xgb.__version__, "| shap:", shap.__version__)
'''
)

md(
    """
## Step 1 - Load and inspect the data

We check the shape, columns, missing values, duplicates, and target values. The
target is cleaned to lowercase, then three target columns are created: the main
binary `is_dropout`, the optional three-class `target_class`, and `is_graduate`.
"""
)

code(
    r"""
df = pd.read_csv(LOCAL_DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
display(df.head())
print("\nMissing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nTarget distribution:")
print(df["Target"].value_counts(dropna=False))

df["Target"] = df["Target"].astype(str).str.strip().str.lower()
valid_targets = {"graduate", "dropout", "enrolled"}
unexpected = sorted(set(df["Target"].unique()) - valid_targets)
assert not unexpected, f"Unexpected target values: {unexpected}"

df = (
    df.replace([np.inf, -np.inf], np.nan)
      .drop_duplicates()
      .reset_index(drop=True)
)

df["is_dropout"] = (df["Target"] == "dropout").astype(int)
df["target_class"] = df["Target"].map(
    {"dropout": 0, "enrolled": 1, "graduate": 2}
)
df["is_graduate"] = (df["Target"] == "graduate").astype(int)

print("\nCleaned shape:", df.shape)
print("\nBinary target:")
print(df["is_dropout"].value_counts().rename({0: "Not dropout", 1: "Dropout"}))
print("\nThree-class target:")
print(
    df["target_class"]
      .value_counts()
      .sort_index()
      .rename({0: "Dropout", 1: "Enrolled", 2: "Graduate"})
)
"""
)

md(
    """
## Step 2 - Feature engineering module

The CSV uses numeric codes for categories such as `Course` and `Application
mode`, so those are kept separate from genuinely numeric columns. This module
contains the feature groups, the semester-level feature formulas, the category
mapping, and human-readable labels. It is written to `src/` so the same logic is
reused by the notebook, the prediction module, and the local app.
"""
)

code("%%writefile src/feature_engineering.py\n" + FEATURE_ENGINEERING)

code(
    r"""
from src.feature_engineering import (
    TARGET_COLS,
    CATEGORICAL_CODE_COLS,
    BINARY_FLAG_COLS,
    MACRO_COLS,
    engineer_features,
    get_feature_columns,
    build_category_map,
    pretty_name,
    binary_options,
)

RAW_FEATURES = [column for column in df.columns if column not in TARGET_COLS]
assert not set(RAW_FEATURES) & set(TARGET_COLS), "Target leakage detected."
assert all(
    column in df.columns
    for column in CATEGORICAL_CODE_COLS + BINARY_FLAG_COLS + MACRO_COLS
)

print("Raw features:", len(RAW_FEATURES))
print("Categorical codes:", len(CATEGORICAL_CODE_COLS))
print("Binary flags:", len(BINARY_FLAG_COLS))
print("Macro-economic:", len(MACRO_COLS))
print(
    "Other numeric:",
    len(RAW_FEATURES)
    - len(CATEGORICAL_CODE_COLS)
    - len(BINARY_FLAG_COLS)
    - len(MACRO_COLS),
)

df_fe = engineer_features(df)
FEATURE_COLUMNS = get_feature_columns(df_fe)
ENGINEERED_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in RAW_FEATURES
]
CATEGORICAL_FEATURES = [
    column for column in CATEGORICAL_CODE_COLS if column in FEATURE_COLUMNS
]
NUMERIC_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]
CATEGORY_MAP = build_category_map(FEATURE_COLUMNS)

assert not set(FEATURE_COLUMNS) & set(TARGET_COLS)
assert set(FEATURE_COLUMNS) == set(RAW_FEATURES) | set(ENGINEERED_FEATURES)
assert all(column in CATEGORY_MAP for column in FEATURE_COLUMNS)

print("\nTotal features:", len(FEATURE_COLUMNS))
print("Engineered features:", len(ENGINEERED_FEATURES))
print("\nEngineered feature names:")
print(ENGINEERED_FEATURES)
print("\nCategory counts:")
print(pd.Series(CATEGORY_MAP).value_counts())
display(df_fe[ENGINEERED_FEATURES].describe().T.round(2))
"""
)

md(
    """
## Step 3 - Train / validation / test split

We use a stratified 70/15/15 split on `is_dropout`. The validation set is for
early stopping and threshold selection. The test set stays untouched until the
final evaluation.
"""
)

code(
    r"""
X = df_fe[FEATURE_COLUMNS].copy()
y = df_fe["is_dropout"].copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

assert len(set(X_train.index) & set(X_valid.index)) == 0
assert len(set(X_train.index) & set(X_test.index)) == 0
assert len(set(X_valid.index) & set(X_test.index)) == 0

print(f"Train: {X_train.shape} | dropout rate: {y_train.mean():.3f}")
print(f"Valid: {X_valid.shape} | dropout rate: {y_valid.mean():.3f}")
print(f"Test:  {X_test.shape} | dropout rate: {y_test.mean():.3f}")
"""
)

md(
    """
## Step 4 - Preprocessing module

Categorical codes use frequency encoding learned from the training set. Missing
numeric values use the training median. No scaling is needed for XGBoost. The
fitted object is a plain dictionary, which makes it easy to save and load in the
notebook, the prediction module, and Streamlit.
"""
)

code("%%writefile src/preprocessing.py\n" + PREPROCESSING)

code(
    r"""
from src.preprocessing import fit_preprocessor, transform_features

preprocessor = fit_preprocessor(
    X_train,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
)

X_train_p = transform_features(X_train, preprocessor)
X_valid_p = transform_features(X_valid, preprocessor)
X_test_p = transform_features(X_test, preprocessor)

assert not X_train_p.isna().any().any()
assert not X_valid_p.isna().any().any()
assert not X_test_p.isna().any().any()
assert not np.isinf(X_train_p.to_numpy()).any()

joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

print("Encoded shapes:", X_train_p.shape, X_valid_p.shape, X_test_p.shape)
print("\nTop course frequencies (learned from train only):")
print(
    pd.Series(preprocessor["frequency_maps"]["Course"])
      .sort_values(ascending=False)
      .head()
)
print("\nMissing values after preprocessing:", int(X_train_p.isna().sum().sum()))
"""
)

md(
    """
## Step 5 - Train XGBoost

The binary target is not perfectly balanced, so `scale_pos_weight` is calculated
from the training split. Early stopping uses the validation AUC. The test set is
not involved.
"""
)

code(
    r"""
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    early_stopping_rounds=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train_p,
    y_train,
    eval_set=[(X_valid_p, y_valid)],
    verbose=False,
)

valid_prob = model.predict_proba(X_valid_p)[:, 1]
test_prob = model.predict_proba(X_test_p)[:, 1]

print("scale_pos_weight:", round(scale_pos_weight, 3))
print("Best iteration:", model.best_iteration)
print("Best validation AUC:", round(float(model.best_score), 4))
"""
)

md(
    """
## Step 6 - Choose a decision threshold on validation data

Youden's J (`TPR - FPR`) picks the threshold that balances catching at-risk
students against false alarms. The threshold is saved with the model.
"""
)

code(
    r"""
fpr, tpr, thresholds = roc_curve(y_valid, valid_prob)
youden_j = tpr - fpr
best_index = int(np.argmax(youden_j))
best_threshold = float(thresholds[best_index])

if not np.isfinite(best_threshold) or best_threshold <= 0 or best_threshold >= 1:
    best_threshold = 0.5


def risk_level(probability):
    for level, (lower, upper) in RISK_BANDS.items():
        if lower <= probability < upper:
            return level
    return "Critical"


print(f"Best threshold (Youden's J): {best_threshold:.4f}")
print(f"Validation Youden's J: {youden_j[best_index]:.4f}")
print("Validation confusion matrix at tuned threshold:")
print(confusion_matrix(y_valid, (valid_prob >= best_threshold).astype(int)))
"""
)

md(
    """
## Step 7 - Evaluate on validation and test

Recall for the dropout class is the key metric for early warning, but precision
is reported as well. Metrics at the tuned threshold and at the default 0.50
threshold are both saved.
"""
)

code(
    r"""
def to_python(obj):
    if isinstance(obj, dict):
        return {str(key): to_python(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "split": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["Not dropout", "Dropout"],
            output_dict=True,
            zero_division=0,
        ),
    }
    print(f"\n{name.upper()} (threshold={threshold:.4f})")
    print(f"  Accuracy : {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall   : {result['recall']:.4f}")
    print(f"  F1       : {result['f1']:.4f}")
    print(f"  ROC AUC  : {result['roc_auc']:.4f}")
    print(f"  PR AUC   : {result['pr_auc']:.4f}")
    print("  Confusion matrix:")
    print(np.array(result["confusion_matrix"]))
    return result


validation_metrics = evaluate_split(
    "Validation", y_valid, valid_prob, best_threshold
)
test_metrics = evaluate_split("Test", y_test, test_prob, best_threshold)
validation_at_05 = evaluate_split(
    "Validation at 0.50", y_valid, valid_prob, 0.50
)
test_at_05 = evaluate_split("Test at 0.50", y_test, test_prob, 0.50)

evaluation_report = {
    "dataset": {
        "path": str(DATASET_PATH),
        "shape": list(df.shape),
        "target_distribution": df["Target"].value_counts().to_dict(),
    },
    "features": {
        "count": len(FEATURE_COLUMNS),
        "categorical_encoded": len(CATEGORICAL_FEATURES),
        "numeric": len(NUMERIC_FEATURES),
        "engineered": len(ENGINEERED_FEATURES),
    },
    "split": {
        "train": list(X_train.shape),
        "valid": list(X_valid.shape),
        "test": list(X_test.shape),
    },
    "threshold_selection": {
        "method": "Youden's J statistic on validation set",
        "best_threshold": best_threshold,
        "youden_j": float(youden_j[best_index]),
    },
    "validation": validation_metrics,
    "test": test_metrics,
    "validation_at_0_5": validation_at_05,
    "test_at_0_5": test_at_05,
    "risk_bands": RISK_BANDS,
}

with open(OUTPUTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as file:
    json.dump(to_python(evaluation_report), file, indent=2)

print("\nSaved:", OUTPUTS_DIR / "evaluation_report.json")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, title, actual, probabilities in zip(
    axes,
    ["Validation", "Test"],
    [y_valid, y_test],
    [valid_prob, test_prob],
):
    matrix = confusion_matrix(actual, (probabilities >= best_threshold).astype(int))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axis,
        xticklabels=["Not dropout", "Dropout"],
        yticklabels=["Not dropout", "Dropout"],
    )
    axis.set_title(f"{title} confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
plt.tight_layout()
plt.show()
"""
)

md(
    """
## Step 8 - XGBoost feature importance

This is the model's own split-based importance. It is useful as a quick ranking,
while SHAP gives the directional and individual explanations.
"""
)

code(
    r"""
importance_df = (
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

print(importance_df.head(15).to_string(index=False))

top_importance = importance_df.head(20).sort_values("importance")
plt.figure(figsize=(10, 7))
plt.barh(top_importance["feature"], top_importance["importance"], color="#5c6bc0")
plt.xlabel("XGBoost importance")
plt.title("Top 20 features")
plt.tight_layout()
plt.show()
"""
)

md(
    """
## Step 9 - Global SHAP and category contribution

SHAP values are computed on the test sample. The global ranking uses mean
absolute SHAP, and the category view groups those values into Academic,
Engagement, Financial, Demographic, Institutional/Course, and Macro-Economic.
"""
)

code(
    r"""
EXPLAINER = shap.TreeExplainer(model)
shap_sample = (
    X_test_p
    if len(X_test_p) <= 1000
    else X_test_p.sample(1000, random_state=RANDOM_STATE)
)
shap_values = EXPLAINER.shap_values(shap_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.DataFrame(
    {"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs_shap}
)
global_importance["percent_contribution"] = (
    100
    * global_importance["mean_abs_shap"]
    / global_importance["mean_abs_shap"].sum()
)
global_importance = (
    global_importance
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
global_importance.to_csv(
    OUTPUTS_DIR / "shap_global_importance.csv", index=False
)

global_importance["category"] = global_importance["feature"].map(CATEGORY_MAP)
category_contribution = (
    global_importance.groupby("category", as_index=False)["mean_abs_shap"]
    .sum()
)
category_contribution["percent_contribution"] = (
    100
    * category_contribution["mean_abs_shap"]
    / category_contribution["mean_abs_shap"].sum()
)
category_contribution = (
    category_contribution
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
category_contribution.to_csv(
    OUTPUTS_DIR / "category_contribution.csv", index=False
)

print(global_importance.head(15).to_string(index=False))
print("\nCategory contribution:")
print(
    category_contribution[["category", "percent_contribution"]]
    .to_string(index=False)
)

shap.summary_plot(
    shap_values,
    shap_sample,
    feature_names=FEATURE_COLUMNS,
    max_display=20,
    show=False,
)
plt.title("SHAP global importance (test sample)")
plt.tight_layout()
plt.show()
"""
)

md(
    """
## Step 10 - Save the model artifacts

The model, feature list, threshold, category map, metadata, and default input
values are saved. Defaults are training medians for numeric fields and training
modes for encoded categorical and binary fields. The app uses them when a field
is missing.
"""
)

code(
    r"""
joblib.dump(model, MODELS_DIR / "dropout_model.joblib")

with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
    json.dump(FEATURE_COLUMNS, file, indent=2)

with open(MODELS_DIR / "threshold.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "best_threshold": best_threshold,
            "method": "Youden's J statistic on validation set",
            "youden_j": float(youden_j[best_index]),
            "risk_bands": RISK_BANDS,
        },
        file,
        indent=2,
    )

with open(MODELS_DIR / "category_map.json", "w", encoding="utf-8") as file:
    json.dump(CATEGORY_MAP, file, indent=2)

train_raw = df.loc[X_train.index, RAW_FEATURES]
default_input_values = {}
for column in RAW_FEATURES:
    if column in CATEGORICAL_CODE_COLS or column in BINARY_FLAG_COLS:
        value = train_raw[column].mode(dropna=True).iloc[0]
        default_input_values[column] = int(value)
    else:
        value = pd.to_numeric(train_raw[column], errors="coerce").median()
        default_input_values[column] = float(value) if pd.notna(value) else 0.0

with open(
    MODELS_DIR / "default_input_values.json", "w", encoding="utf-8"
) as file:
    json.dump(default_input_values, file, indent=2)

metadata = {
    "model_type": "xgboost",
    "target": "is_dropout",
    "random_state": RANDOM_STATE,
    "feature_count": len(FEATURE_COLUMNS),
    "train_shape": list(X_train.shape),
    "valid_shape": list(X_valid.shape),
    "test_shape": list(X_test.shape),
    "best_threshold": best_threshold,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "dataset_shape": list(df.shape),
    "xgboost_version": xgb.__version__,
    "shap_version": shap.__version__,
    "test_metrics": {
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
    },
}
with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("Saved model artifacts:")
for path in sorted(MODELS_DIR.glob("*")):
    print(" -", path.name)
"""
)

md(
    """
## Step 11 - Prediction and explanation module

This module loads the artifacts, fills missing fields with training defaults,
engineers features, predicts, computes individual SHAP values, and builds the
recommendation list. The same module is used by the Kaggle assistant and by the
local Streamlit app.
"""
)

code("%%writefile src/predict.py\n" + PREDICT)

code(
    r"""
from src.predict import DropoutPredictor

predictor = DropoutPredictor(models_dir=MODELS_DIR)

default_result = predictor.predict(default_input_values)
print("Default sample student")
print(json.dumps(default_result, indent=2, default=str)[:2500])

high_risk_student = dict(default_input_values)
high_risk_student.update(
    {
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 0,
        "Curricular units 1st sem (grade)": 0.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 0,
        "Curricular units 2nd sem (grade)": 0.0,
        "Debtor": 1,
        "Tuition fees up to date": 0,
        "Scholarship holder": 0,
        "Admission grade": 95.0,
    }
)

low_risk_student = dict(default_input_values)
low_risk_student.update(
    {
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 16.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 16.0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Scholarship holder": 1,
        "Admission grade": 150.0,
    }
)

high_result = predictor.predict(high_risk_student)
low_result = predictor.predict(low_risk_student)

for name, result in [
    ("High-risk example", high_result),
    ("Low-risk example", low_result),
]:
    print(
        f"\n{name}: {result['dropout_probability_percent']}% "
        f"({result['risk_level']} risk)"
    )
    print("Top reasons:")
    for item in result["top_reasons"]:
        print(
            f"  - {item['reason']} | {item['category']} | "
            f"{item['percent_contribution']:.1f}%"
        )

assert high_result["dropout_probability"] > low_result["dropout_probability"]

with open(
    OUTPUTS_DIR / "example_prediction.json", "w", encoding="utf-8"
) as file:
    json.dump(high_result, file, indent=2, default=str)

print("\nSaved:", OUTPUTS_DIR / "example_prediction.json")
print("\nPrediction module checks passed.")
"""
)

md(
    """
## Step 12 - Kaggle chatbot (ipywidgets)

Fill the form, click **Predict Risk**, and the assistant shows the probability,
risk level, key factors, category chart, and recommended actions. The chat box
accepts `predict`, `use default student`, `explain risk`, and
`show recommendations`. Kaggle does not render Streamlit inline, so this
ipywidgets version is the in-notebook chatbot; the Streamlit app is exported for
the later local shift.
"""
)

code(
    r'''
FORM_SECTIONS = {
    "Basic Information": [
        "Marital status", "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification", "Nacionality",
        "Gender", "Age at enrollment", "International", "Displaced",
        "Educational special needs",
    ],
    "Family Background": [
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ],
    "Academic Entry": [
        "Previous qualification (grade)", "Admission grade",
    ],
    "Financial Information": [
        "Debtor", "Tuition fees up to date", "Scholarship holder",
    ],
    "Semester 1 Performance": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
    "Semester 2 Performance": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "Economic Indicators": [
        "Unemployment rate", "Inflation rate", "GDP",
    ],
}

RISK_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}

form_widgets = {}
section_boxes = []

for section, fields in FORM_SECTIONS.items():
    rows = []
    for field in fields:
        default = default_input_values[field]
        if field in BINARY_FLAG_COLS:
            widget = widgets.Dropdown(
                options=binary_options(field), value=int(default)
            )
        elif field in CATEGORICAL_CODE_COLS or field == "Age at enrollment":
            widget = widgets.IntText(value=int(default))
        else:
            widget = widgets.FloatText(value=float(default))
        widget.layout = widgets.Layout(width="130px")
        form_widgets[field] = widget
        label = widgets.HTML(
            "<div style='width:285px;font-size:12px;line-height:1.2'>"
            f"{pretty_name(field)}</div>"
        )
        rows.append(
            widgets.HBox(
                [label, widget],
                layout=widgets.Layout(align_items="center", margin="2px 0"),
            )
        )
    section_boxes.append(widgets.VBox(rows))

accordion = widgets.Accordion(
    children=section_boxes, titles=list(FORM_SECTIONS.keys())
)
accordion.selected_index = 0

predict_button = widgets.Button(description="Predict Risk", button_style="success")
default_button = widgets.Button(description="Use Default Student")
high_button = widgets.Button(description="High-Risk Example")
low_button = widgets.Button(description="Low-Risk Example")
output = widgets.Output()
last_result = {"value": None}


def read_form():
    return {field: widget.value for field, widget in form_widgets.items()}


def write_form(values):
    for field, widget in form_widgets.items():
        widget.value = values.get(field, default_input_values[field])


def display_result(result):
    last_result["value"] = result
    output.clear_output()
    with output:
        color = RISK_COLORS.get(result["risk_level"], "#555555")
        display(
            HTML(
                f"""
                <div style="border:1px solid #ddd;border-radius:12px;padding:16px;
                            margin:8px 0;background:#fafafa">
                    <div style="font-size:13px;color:#666">Estimated dropout probability</div>
                    <div style="font-size:38px;font-weight:700">{result['dropout_probability_percent']}%</div>
                    <div style="font-size:18px;color:{color};font-weight:600">
                        Risk level: {result['risk_level']}
                    </div>
                    <div style="font-size:12px;color:#777">
                        Decision threshold: {result['threshold_used']:.2f}
                    </div>
                </div>
                """
            )
        )
        reason_rows = "".join(
            f"<tr><td>{index}</td><td>{item['reason']}</td>"
            f"<td>{item['category']}</td>"
            f"<td>{item['percent_contribution']:.1f}%</td>"
            f"<td>{item['direction']}</td></tr>"
            for index, item in enumerate(result["top_reasons"], 1)
        )
        display(
            HTML(
                f"""
                <h3 style="margin:10px 0 6px">Key factors</h3>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <tr style="background:#eef1f7">
                        <th align="left">#</th><th align="left">Reason</th>
                        <th align="left">Category</th>
                        <th align="left">Contribution</th>
                        <th align="left">Direction</th>
                    </tr>
                    {reason_rows}
                </table>
                """
            )
        )
        category_df = pd.DataFrame(result["category_contributions"])
        if not category_df.empty:
            axis = (
                category_df.set_index("category")["percent_contribution"]
                .plot(kind="barh", figsize=(7, 3), color="#5c6bc0")
            )
            axis.invert_yaxis()
            axis.set_xlabel("Share of explanation (%)")
            axis.set_ylabel("")
            plt.tight_layout()
            plt.show()
            plt.close()
        display(
            HTML(
                "<h3 style='margin:10px 0 6px'>Recommended actions</h3><ul>"
                + "".join(
                    f"<li>{item}</li>" for item in result["recommendations"]
                )
                + "</ul>"
            )
        )
        if result["missing_fields_filled"]:
            filled = ", ".join(
                pretty_name(field)
                for field in result["missing_fields_filled"]
            )
            display(
                HTML(
                    "<p style='color:#ef6c00;font-size:12px'>"
                    f"Filled missing values with training defaults: {filled}</p>"
                )
            )
        display(
            HTML(
                f"<p style='color:#777;font-size:12px'>{result['disclaimer']}</p>"
            )
        )


def on_predict(_):
    display_result(predictor.predict(read_form()))


def on_default(_):
    write_form(default_input_values)
    display_result(predictor.predict(default_input_values))


def on_high(_):
    write_form(high_risk_student)
    display_result(predictor.predict(high_risk_student))


def on_low(_):
    write_form(low_risk_student)
    display_result(predictor.predict(low_risk_student))


for button, callback in [
    (predict_button, on_predict),
    (default_button, on_default),
    (high_button, on_high),
    (low_button, on_low),
]:
    button.on_click(callback)

chat_input = widgets.Text(
    placeholder="Try: predict | use default student | explain risk | show recommendations",
    layout=widgets.Layout(width="560px"),
)
chat_send = widgets.Button(description="Send")
chat_output = widgets.Output()


def on_chat(_):
    text = chat_input.value.strip()
    if not text:
        return
    command = text.lower()
    chat_input.value = ""
    reply = ""

    if command in {"predict", "predict risk"}:
        result = predictor.predict(read_form())
        display_result(result)
        reply = (
            f"Estimated dropout probability: "
            f"{result['dropout_probability_percent']}% "
            f"({result['risk_level']} risk)."
        )
    elif command in {"use default student", "default"}:
        write_form(default_input_values)
        result = predictor.predict(default_input_values)
        display_result(result)
        reply = (
            "Loaded the default student. "
            f"Estimated probability: {result['dropout_probability_percent']}% "
            f"({result['risk_level']} risk)."
        )
    elif command in {"explain risk", "explain"}:
        result = last_result["value"]
        if result:
            reply = "<br>".join(
                f"{index}. {item['reason']} "
                f"({item['category']}, {item['percent_contribution']:.1f}%)"
                for index, item in enumerate(result["top_reasons"][:5], 1)
            )
        else:
            reply = "Please run a prediction first."
    elif command in {"show recommendations", "recommendations"}:
        result = last_result["value"]
        reply = (
            "<br>".join(f"- {item}" for item in result["recommendations"])
            if result
            else "Please run a prediction first."
        )
    else:
        reply = (
            "Try: predict, use default student, explain risk, "
            "show recommendations."
        )

    with chat_output:
        display(HTML(f"<b>You:</b> {text}"))
        display(HTML(f"<b>Assistant:</b> {reply}"))


chat_send.on_click(on_chat)
chat_input.on_submit(on_chat)

display(
    HTML(
        "<h3>Student Dropout Risk Assistant</h3>"
        "<p>Enter student information, then click <b>Predict Risk</b>. "
        "The assistant estimates risk and explains the main factors.</p>"
    )
)
display(widgets.HBox([predict_button, default_button, high_button, low_button]))
display(accordion)
display(output)
display(HTML("<h4>Ask the assistant</h4>"))
display(widgets.HBox([chat_input, chat_send]))
display(chat_output)
'''
)

md(
    """
## Step 13 - Export the Streamlit app and download the project

The app loads the saved artifacts and never retrains. It includes the same
prediction, SHAP explanation, recommendation, and chat-command behavior in a
local Streamlit interface. This cell writes the app, README, requirements, and a
ZIP of the whole project.
"""
)

code("%%writefile app/chatbot_app.py\n" + APP)

code(
    "readme_text = " + json.dumps(README) + "\n"
    "requirements_text = " + json.dumps(REQUIREMENTS) + "\n"
    "\n"
    "(PROJECT_DIR / 'README.md').write_text(readme_text, encoding='utf-8')\n"
    "(PROJECT_DIR / 'requirements.txt').write_text(requirements_text, encoding='utf-8')\n"
    "\n"
    "zip_path = (\n"
    "    Path('/kaggle/working/student_dropout_project.zip')\n"
    "    if Path('/kaggle/working').exists()\n"
    "    else PROJECT_DIR.parent / 'student_dropout_project.zip'\n"
    ")\n"
    "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:\n"
    "    for file in PROJECT_DIR.rglob('*'):\n"
    "        if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':\n"
    "            archive.write(file, file.relative_to(PROJECT_DIR))\n"
    "\n"
    "print('ZIP created:', zip_path)\n"
    "print('\\nProject files:')\n"
    "for file in sorted(PROJECT_DIR.rglob('*')):\n"
    "    if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':\n"
    "        print(' -', file.relative_to(PROJECT_DIR))\n"
)

md(
    """
## Done

You now have:

- a trained and threshold-tuned XGBoost model;
- a leakage-safe preprocessing object;
- validation and test metrics;
- global and individual SHAP explanations;
- a working Kaggle risk assistant;
- a Streamlit app and the full project zip for the local shift.

Download `student_dropout_project.zip` from `/kaggle/working` to keep the model,
outputs, source modules, and app. When you move locally, unzip it and run:

```bash
pip install -r requirements.txt
streamlit run app/chatbot_app.py
```

Use this system for support, not punishment. Human review is required.
"""
)


def cell_source(text):
    return text.splitlines(keepends=True)


notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for index, (cell_type, source) in enumerate(cells):
    cell = {
        "cell_type": cell_type,
        "id": f"cell-{index:02d}",
        "metadata": {},
        "source": cell_source(source),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

out = BASE / "Student_Dropout_Kaggle.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out, "with", len(cells), "cells")
