# Kaggle Guide - Student Dropout Prediction and Explainable Assistant

This guide is for the ready-to-import notebook:

```text
kaggle/Student_Dropout_Kaggle.ipynb
```

The notebook is self-contained: it writes the `src/` modules, trains the model,
saves the artifacts, runs the Kaggle chatbot, and exports the local Streamlit
app plus a ZIP of the whole project.

---

## 1. What you need

1. A Kaggle account.
2. The dataset file:

   ```text
   dataset_outliers_removed_balanced.csv
   ```

3. The notebook file:

   ```text
   Student_Dropout_Kaggle.ipynb
   ```

The CSV must keep that exact filename. The notebook searches `/kaggle/input`
recursively by filename, so the Kaggle dataset title/slug can be anything.

---

## 2. Upload the dataset to Kaggle

1. Open Kaggle and click **Datasets** in the left sidebar.
2. Click **New Dataset**.
3. Upload `dataset_outliers_removed_balanced.csv`.
4. Give the dataset a title, for example `student-dropout-balanced`.
5. Click **Create**.

You do not need to change the CSV. The notebook handles target cleaning,
duplicate detection, feature engineering, and preprocessing.

---

## 3. Import the notebook

1. Open Kaggle and click **Code** in the left sidebar.
2. Click **New Notebook**.
3. In the notebook menu, choose **File > Import Notebook**.
4. Upload `Student_Dropout_Kaggle.ipynb`.
5. In the right sidebar, open **Data > Your Datasets** and add the dataset you
   created in step 2.

If the import option is unavailable, create a blank notebook and copy the cells
in order. Each code cell is complete; do not skip the `%%writefile` cells.

---

## 4. Run the notebook

1. Set the accelerator to **None** (CPU). The model is small.
2. Click **Run all**.
3. Wait for all cells to finish. On a normal Kaggle CPU, training and SHAP
   usually take a couple of minutes.

Expected progress:

| Step | What happens | Main output |
|---|---|---|
| 0 | Imports, folders, dataset discovery | Project path and library versions |
| 1 | Load and inspect the CSV | Shape, target distribution |
| 2 | Write and use feature engineering module | 27 engineered features |
| 3 | Split 70/15/15 | Three disjoint splits |
| 4 | Fit preprocessing on train only | `models/preprocessor.joblib` |
| 5 | Train XGBoost | Best iteration and validation AUC |
| 6 | Tune threshold on validation | `models/threshold.json` |
| 7 | Evaluate validation and test | `outputs/evaluation_report.json` |
| 8 | XGBoost feature importance | `outputs/feature_importance.csv` |
| 9 | Global SHAP + category contribution | SHAP CSV files |
| 10 | Save model artifacts | `models/*.joblib`, `models/*.json` |
| 11 | Write and test prediction module | `src/predict.py`, example prediction |
| 12 | Build Kaggle chatbot | ipywidgets form + chat |
| 13 | Export Streamlit app and ZIP | `student_dropout_project.zip` |

The test split is used only in step 7 for the final evaluation. The threshold is
selected on validation data only, and preprocessing is fitted on training data
only.

---

## 5. Use the Kaggle chatbot

Scroll to **Step 12**. You will see:

- A collapsible form with these sections:
  - Basic Information
  - Family Background
  - Academic Entry
  - Financial Information
  - Semester 1 Performance
  - Semester 2 Performance
  - Economic Indicators
- Four buttons:
  - **Predict Risk**
  - **Use Default Student**
  - **High-Risk Example**
  - **Low-Risk Example**
- A chat box.

### Buttons

1. Click **High-Risk Example** to load a student with low approval, failed units,
   outstanding debt, and unpaid tuition.
2. Click **Predict Risk**.
3. Read the probability, risk level, key factors, category chart, and
   recommended actions.
4. Click **Low-Risk Example** and repeat to compare.

### Chat commands

Type one of these in the chat box and press **Enter** (or click **Send**):

```text
predict
use default student
explain risk
show recommendations
```

The chat assistant uses the same prediction and SHAP explanation code as the
form. It is an in-notebook assistant, not a separate web server.

---

## 6. Download the outputs

The notebook writes the project to:

```text
/kaggle/working/student_dropout_project
```

and creates:

```text
/kaggle/working/student_dropout_project.zip
```

Download the ZIP from either:

- the **Output** panel on the right side of the notebook, or
- the Kaggle file browser for `/kaggle/working`.

The ZIP contains:

```text
data/
models/
outputs/
src/
app/
README.md
requirements.txt
```

Important files:

```text
models/dropout_model.joblib
models/preprocessor.joblib
models/feature_columns.json
models/threshold.json
models/metadata.json
models/category_map.json
models/default_input_values.json

outputs/evaluation_report.json
outputs/feature_importance.csv
outputs/shap_global_importance.csv
outputs/category_contribution.csv
outputs/example_prediction.json

src/feature_engineering.py
src/preprocessing.py
src/predict.py

app/chatbot_app.py
```

---

## 7. Expected results

The exact numbers can vary slightly with library versions, but the current
dataset produces approximately:

```text
Validation ROC-AUC: 0.93
Test ROC-AUC:       0.91
Test dropout recall: 0.85
Tuned threshold:    ~0.32
```

The target is imbalanced (Graduate 2051, Dropout 1273, Enrolled 747), so
`scale_pos_weight` is used for the binary target. Do not expect a balanced
accuracy of 0.95; recall for the dropout class and ROC-AUC matter more for early
warning.

Risk bands are fixed:

```text
Low:      probability < 0.35
Medium:   0.35 <= probability < 0.65
High:     0.65 <= probability < 0.85
Critical: probability >= 0.85
```

The tuned threshold is used for the binary `predicted_dropout` flag. Risk levels
use the fixed bands above, so a student can be "Medium" risk while the binary
threshold is different from the band boundary.

---

## 8. Troubleshooting

### `FileNotFoundError: Add dataset_outliers_removed_balanced.csv...`

The notebook did not find the CSV under `/kaggle/input`.

- Confirm the file is added to the notebook from **Data > Your Datasets**.
- Confirm the filename is exactly `dataset_outliers_removed_balanced.csv`.
- If Kaggle renamed the file, rename it back before creating the dataset.
- Rerun the setup cell after adding the dataset.

### `ModuleNotFoundError: xgboost`, `shap`, `seaborn`, or `ipywidgets`

Kaggle images normally include these. If your image does not, turn on internet in
the notebook settings and run:

```python
!pip install xgboost shap seaborn ipywidgets
```

Then rerun from the setup cell.

### The widget form does not appear

- Make sure you ran the cell, not just selected it.
- Kaggle renders ipywidgets in notebook output; a static notebook preview will
  not show them.
- Refresh the notebook and rerun the widget cell.
- If widgets are still blocked, the rest of the project still works; use the
  Streamlit app locally after downloading the ZIP.

### SHAP plot is blank or slow

The notebook samples at most 1,000 test rows for global SHAP. On a slow CPU,
wait for the cell to finish. If the plot is still blank, rerun just the SHAP
cell; the CSV files are saved before the plot.

### The ZIP is missing files

Run the final export cell after all previous cells. It zips the current project
folder, so any cell that was skipped will leave its output missing.

### I want to change the decision threshold

Edit the threshold after step 6 only if you have a clear policy reason. The saved
`models/threshold.json` is used by the app. Do not tune on the test set.

---

## 9. Moving to local Streamlit

Streamlit does not render inline inside Kaggle. For the local shift:

1. Download and unzip `student_dropout_project.zip`.
2. Open a terminal in the unzipped folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The app uses the current Streamlit `width` API; use Streamlit 1.50 or newer.

4. Start the app:

   ```bash
   streamlit run app/chatbot_app.py
   ```

5. Open the local URL printed by Streamlit.

The app loads the saved model and preprocessor. It never retrains. The form and
chat commands use the same `src/predict.py` module as the Kaggle notebook.

---

## 10. Ethical use

This tool estimates risk from historical patterns. It does not establish causes
and should not be used as the sole basis for denying admission, scholarships, or
other opportunities.

- Use it for support, not punishment.
- Require human review before any action.
- Do not enter names, student IDs, or other direct identifiers.
- Check for bias across demographic groups before real deployment.
- Treat explanations as conversation starters for advisors and counselors.
