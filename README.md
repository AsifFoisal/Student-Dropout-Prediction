# Student Dropout Prediction and Explainable Assistant

An end-to-end, leakage-safe machine-learning project that estimates a student's
dropout risk, explains the main factors behind the estimate, and suggests
supportive actions. It is built for **Kaggle first** and can be moved to a local
Streamlit app later.

> This is a decision-support tool, not an automatic decision. Human review is
> required before any academic or financial action is taken.

## What the project does

1. Loads `dataset_outliers_removed_balanced.csv`.
2. Creates a binary `is_dropout` target and optional three-class targets.
3. Engineers semester-level academic features (approval rates, failed units,
   effective grades, semester changes, risk flags, evaluation ratios).
4. Fits frequency encoding and median imputation on the **training split only**.
5. Trains an XGBoost classifier with early stopping on a validation split.
6. Selects a decision threshold on validation data using Youden's J.
7. Evaluates on a held-out test set (accuracy, precision, recall, F1, ROC AUC,
   PR AUC, confusion matrix).
8. Produces global SHAP importance, category-level contributions, and individual
   explanations.
9. Saves the model, preprocessor, feature list, threshold, category map, default
   input values, and metadata.
10. Provides a Kaggle ipywidgets assistant and a local Streamlit chatbot.

## Dataset

The project expects this file:

```text
dataset_outliers_removed_balanced.csv
```

Expected columns include the 36 UCI student-performance features plus `Target`
with values `Graduate`, `Dropout`, or `Enrolled`.

The training notebook uses the full feature set. The app deliberately asks for
only 14 high-signal fields: course, age, gender, displaced status, admission
grade, semester 1/2 enrolled units, approved units, grades, tuition-fee status,
debt status, and scholarship status. Course names, female/male, and no/yes
choices are shown as text instead of numeric codes. Background details such as
marital status, nationality, family occupation, application mode, and macro
indicators are filled with typical training values and are left out of the
visible key-factor list.

The file name says "balanced", but the version used here has:

```text
Graduate 2051
Dropout  1273
Enrolled  747
```

The binary target is therefore handled with `scale_pos_weight`; no SMOTE is
applied. Outliers are not removed again.

## Kaggle quickstart

1. Create a Kaggle notebook.
2. Add `dataset_outliers_removed_balanced.csv` as a Kaggle dataset (or upload it
   to the notebook's working directory).
3. Import/download `kaggle/Student_Dropout_Kaggle.ipynb` and open it in Kaggle.
4. Click **Run all**.
5. Use the 14-field ipywidgets form at the bottom to make predictions and ask
   the assistant for explanations or recommendations.
6. Download `/kaggle/working/student_dropout_project.zip` from the notebook output.

The notebook searches `/kaggle/input` for the CSV by filename, so the Kaggle
dataset slug can be anything.

## Local quickstart

After downloading and unzipping the project:

```bash
pip install -r requirements.txt
streamlit run app/chatbot_app.py
```

The app loads the saved artifacts and never retrains the model. It asks for the
14 core fields, uses readable labels for coded values, and fills omitted
background fields with typical training values. It was tested with Streamlit
1.66; `requirements.txt` uses 1.50+ because the app uses the current `width`
API.

## Project structure

```text
student_dropout_project/
├── data/
│   └── dataset_outliers_removed_balanced.csv
├── models/
│   ├── dropout_model.joblib
│   ├── preprocessor.joblib
│   ├── feature_columns.json
│   ├── threshold.json
│   ├── metadata.json
│   ├── category_map.json
│   └── default_input_values.json
├── outputs/
│   ├── evaluation_report.json
│   ├── feature_importance.csv
│   ├── shap_global_importance.csv
│   ├── category_contribution.csv
│   └── example_prediction.json
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── predict.py
├── app/
│   └── chatbot_app.py
├── README.md
└── requirements.txt
```

## Notebook flow

The Kaggle notebook is organized as numbered cells with a short explanation
before each step:

1. Setup, imports, dataset discovery.
2. Load and inspect data.
3. Define targets and feature groups.
4. Feature engineering.
5. Train/validation/test split.
6. Preprocessing fitted on train only.
7. XGBoost training with early stopping.
8. Threshold selection on validation.
9. Evaluation on validation and test.
10. XGBoost feature importance.
11. Global SHAP explanation and category contribution.
12. Save model artifacts and default input values.
13. Prediction, SHAP explanation, and recommendation module.
14. Kaggle ipywidgets chatbot with the 14-field form.
15. Export Streamlit app, README, requirements, and ZIP.

## Example input

```python
student = {
    "Course": 9500,                  # Nursing
    "Age at enrollment": 20,
    "Gender": 0,                     # Female
    "Displaced": 0,                  # No
    "Admission grade": 120.0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (approved)": 4,
    "Curricular units 1st sem (grade)": 12.0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (approved)": 3,
    "Curricular units 2nd sem (grade)": 11.0,
    "Tuition fees up to date": 1,    # Yes
    "Debtor": 0,                     # No
    "Scholarship holder": 0,         # No
}
```

Any omitted field is filled with a training median or mode. The app passes
typical values for the background fields it does not ask for and filters those
fields out of the visible explanation.

## Example output shape

```json
{
  "dropout_probability": 0.78,
  "risk_level": "High",
  "predicted_dropout": 1,
  "top_reasons": [
    {
      "feature": "overall_approval_rate",
      "category": "Engagement",
      "value": 0.22,
      "shap_value": 0.18,
      "percent_contribution": 21.5,
      "direction": "increases risk",
      "reason": "Low overall approval rate (22%)"
    }
  ],
  "category_contributions": [
    {"category": "Engagement", "percent_contribution": 46.2}
  ],
  "recommendations": [
    "Contact the student for a supportive check-in.",
    "Create a study recovery plan."
  ],
  "disclaimer": "This prediction is a decision-support tool..."
}
```

## Risk levels

```text
Low:      probability < 0.35
Medium:   0.35 <= probability < 0.65
High:     0.65 <= probability < 0.85
Critical: probability >= 0.85
```

The tuned threshold (selected on validation data) is used for the binary
`predicted_dropout` flag. Risk levels use the fixed bands above.

## Ethical use

- Do not use this model as the sole basis for denying admission, scholarships,
  or other opportunities.
- Do not enter names, student IDs, or other direct identifiers.
- Review explanations with a counselor or academic advisor.
- Monitor for bias across demographic groups before any real deployment.
- The model identifies patterns in historical data; it does not establish causes.

## Reproducibility

- `RANDOM_STATE = 42`
- Stratified 70/15/15 split
- Preprocessing fitted on training data only
- Threshold tuned on validation data only
- Test set used only for final evaluation
