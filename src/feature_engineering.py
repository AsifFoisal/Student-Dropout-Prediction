"""Feature engineering, feature grouping, and human-readable labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLS = ["Target", "is_dropout", "target_class", "is_graduate"]

CATEGORICAL_CODE_COLS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

BINARY_FLAG_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

MACRO_COLS = ["Unemployment rate", "Inflation rate", "GDP"]

FINANCIAL_FEATURES = {"Debtor", "Tuition fees up to date", "Scholarship holder"}

ACADEMIC_FEATURES = {
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    "effective_avg_grade_sem1",
    "effective_avg_grade_sem2",
    "overall_average_grade",
    "grade_difference_sem2_minus_sem1",
}

DEMOGRAPHIC_FEATURES = {
    "Marital status",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
}

INSTITUTIONAL_FEATURES = {
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
}

MACRO_FEATURES = set(MACRO_COLS)

DISPLAY_NAMES = {
    "Marital status": "Marital status",
    "Application mode": "Application mode",
    "Application order": "Application order",
    "Course": "Course",
    "Daytime/evening attendance": "Daytime/evening attendance",
    "Previous qualification": "Previous qualification",
    "Previous qualification (grade)": "Previous qualification grade",
    "Nacionality": "Nationality",
    "Mother's qualification": "Mother's qualification",
    "Father's qualification": "Father's qualification",
    "Mother's occupation": "Mother's occupation",
    "Father's occupation": "Father's occupation",
    "Admission grade": "Admission grade",
    "Displaced": "Displaced student",
    "Educational special needs": "Educational special needs",
    "Debtor": "Outstanding debt",
    "Tuition fees up to date": "Tuition fees up to date",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship holder",
    "Age at enrollment": "Age at enrollment",
    "International": "International student",
    "Curricular units 1st sem (credited)": "Semester 1 credited units",
    "Curricular units 1st sem (enrolled)": "Semester 1 enrolled units",
    "Curricular units 1st sem (evaluations)": "Semester 1 evaluations",
    "Curricular units 1st sem (approved)": "Semester 1 approved units",
    "Curricular units 1st sem (grade)": "Semester 1 grade",
    "Curricular units 1st sem (without evaluations)": "Semester 1 units without evaluations",
    "Curricular units 2nd sem (credited)": "Semester 2 credited units",
    "Curricular units 2nd sem (enrolled)": "Semester 2 enrolled units",
    "Curricular units 2nd sem (evaluations)": "Semester 2 evaluations",
    "Curricular units 2nd sem (approved)": "Semester 2 approved units",
    "Curricular units 2nd sem (grade)": "Semester 2 grade",
    "Curricular units 2nd sem (without evaluations)": "Semester 2 units without evaluations",
    "Unemployment rate": "Unemployment rate",
    "Inflation rate": "Inflation rate",
    "GDP": "GDP",
    "total_enrolled_units": "Total enrolled units",
    "total_approved_units": "Total approved units",
    "total_credited_units": "Total credited units",
    "total_evaluations": "Total evaluations",
    "total_without_evaluations": "Total units without evaluations",
    "failed_units_sem1": "Failed units in semester 1",
    "failed_units_sem2": "Failed units in semester 2",
    "total_failed_units": "Total failed units",
    "approval_rate_sem1": "Semester 1 approval rate",
    "approval_rate_sem2": "Semester 2 approval rate",
    "overall_approval_rate": "Overall approval rate",
    "effective_avg_grade_sem1": "Semester 1 effective grade",
    "effective_avg_grade_sem2": "Semester 2 effective grade",
    "overall_average_grade": "Overall average grade",
    "grade_difference_sem2_minus_sem1": "Grade change from semester 1 to 2",
    "approved_difference_sem2_minus_sem1": "Approved-unit change from semester 1 to 2",
    "approval_rate_difference_sem2_minus_sem1": "Approval-rate change from semester 1 to 2",
    "failed_difference_sem2_minus_sem1": "Failed-unit change from semester 1 to 2",
    "no_approved_sem1": "No approved units in semester 1",
    "no_approved_sem2": "No approved units in semester 2",
    "no_approved_both_semesters": "No approved units in both semesters",
    "low_approval_sem1": "Low approval rate in semester 1",
    "low_approval_sem2": "Low approval rate in semester 2",
    "low_approval_both_semesters": "Low approval rate in both semesters",
    "evaluation_to_enrolled_ratio_sem1": "Semester 1 evaluation-to-enrollment ratio",
    "evaluation_to_enrolled_ratio_sem2": "Semester 2 evaluation-to-enrollment ratio",
    "overall_evaluation_to_enrolled_ratio": "Overall evaluation-to-enrollment ratio",
}


def safe_div(numerator, denominator):
    """Divide two series and return 0 wherever the denominator is zero."""
    num = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator)
    den = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator)
    den = den.replace(0, np.nan)
    result = num / den
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the semester-level academic features used by the model."""
    d = data.copy()

    s1_cred = d["Curricular units 1st sem (credited)"]
    s1_enr = d["Curricular units 1st sem (enrolled)"]
    s1_eval = d["Curricular units 1st sem (evaluations)"]
    s1_app = d["Curricular units 1st sem (approved)"]
    s1_grade = d["Curricular units 1st sem (grade)"]
    s1_noeval = d["Curricular units 1st sem (without evaluations)"]

    s2_cred = d["Curricular units 2nd sem (credited)"]
    s2_enr = d["Curricular units 2nd sem (enrolled)"]
    s2_eval = d["Curricular units 2nd sem (evaluations)"]
    s2_app = d["Curricular units 2nd sem (approved)"]
    s2_grade = d["Curricular units 2nd sem (grade)"]
    s2_noeval = d["Curricular units 2nd sem (without evaluations)"]

    d["total_enrolled_units"] = s1_enr + s2_enr
    d["total_approved_units"] = s1_app + s2_app
    d["total_credited_units"] = s1_cred + s2_cred
    d["total_evaluations"] = s1_eval + s2_eval
    d["total_without_evaluations"] = s1_noeval + s2_noeval

    d["failed_units_sem1"] = (s1_enr - s1_app).clip(lower=0)
    d["failed_units_sem2"] = (s2_enr - s2_app).clip(lower=0)
    d["total_failed_units"] = (
        d["total_enrolled_units"] - d["total_approved_units"]
    ).clip(lower=0)

    d["approval_rate_sem1"] = safe_div(s1_app, s1_enr).clip(0, 1)
    d["approval_rate_sem2"] = safe_div(s2_app, s2_enr).clip(0, 1)
    d["overall_approval_rate"] = safe_div(
        d["total_approved_units"], d["total_enrolled_units"]
    ).clip(0, 1)

    d["effective_avg_grade_sem1"] = np.where(s1_app > 0, s1_grade, 0.0)
    d["effective_avg_grade_sem2"] = np.where(s2_app > 0, s2_grade, 0.0)
    d["overall_average_grade"] = safe_div(
        d["effective_avg_grade_sem1"] * s1_app
        + d["effective_avg_grade_sem2"] * s2_app,
        d["total_approved_units"],
    )

    d["grade_difference_sem2_minus_sem1"] = (
        d["effective_avg_grade_sem2"] - d["effective_avg_grade_sem1"]
    )
    d["approved_difference_sem2_minus_sem1"] = s2_app - s1_app
    d["approval_rate_difference_sem2_minus_sem1"] = (
        d["approval_rate_sem2"] - d["approval_rate_sem1"]
    )
    d["failed_difference_sem2_minus_sem1"] = (
        d["failed_units_sem2"] - d["failed_units_sem1"]
    )

    d["no_approved_sem1"] = (s1_app == 0).astype(int)
    d["no_approved_sem2"] = (s2_app == 0).astype(int)
    d["no_approved_both_semesters"] = (
        (s1_app == 0) & (s2_app == 0)
    ).astype(int)
    d["low_approval_sem1"] = (d["approval_rate_sem1"] < 0.5).astype(int)
    d["low_approval_sem2"] = (d["approval_rate_sem2"] < 0.5).astype(int)
    d["low_approval_both_semesters"] = (
        (d["approval_rate_sem1"] < 0.5) & (d["approval_rate_sem2"] < 0.5)
    ).astype(int)

    d["evaluation_to_enrolled_ratio_sem1"] = safe_div(s1_eval, s1_enr)
    d["evaluation_to_enrolled_ratio_sem2"] = safe_div(s2_eval, s2_enr)
    d["overall_evaluation_to_enrolled_ratio"] = safe_div(
        d["total_evaluations"], d["total_enrolled_units"]
    )

    return d.replace([np.inf, -np.inf], np.nan)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return every non-target column after feature engineering."""
    return [c for c in data.columns if c not in TARGET_COLS]


def build_category_map(feature_columns) -> dict[str, str]:
    """Map each model feature to a human-readable explanation category."""
    category_map = {}
    for feature in feature_columns:
        if feature in FINANCIAL_FEATURES:
            category = "Financial"
        elif feature in ACADEMIC_FEATURES:
            category = "Academic"
        elif feature in DEMOGRAPHIC_FEATURES:
            category = "Demographic"
        elif feature in INSTITUTIONAL_FEATURES:
            category = "Institutional/Course"
        elif feature in MACRO_FEATURES:
            category = "Macro-Economic"
        else:
            category = "Engagement"
        category_map[feature] = category
    return category_map


def pretty_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature.replace("_", " ").strip().capitalize())


FRACTION_FEATURES = {
    "approval_rate_sem1",
    "approval_rate_sem2",
    "overall_approval_rate",
    "approval_rate_difference_sem2_minus_sem1",
    "evaluation_to_enrolled_ratio_sem1",
    "evaluation_to_enrolled_ratio_sem2",
    "overall_evaluation_to_enrolled_ratio",
}

GRADE_FEATURES = {
    "effective_avg_grade_sem1",
    "effective_avg_grade_sem2",
    "overall_average_grade",
    "Admission grade",
    "Previous qualification (grade)",
}

BINARY_LABELS = {
    "Debtor": ("No outstanding debt", "Outstanding debt"),
    "Tuition fees up to date": ("Tuition fees not up to date", "Tuition fees up to date"),
    "Scholarship holder": ("No scholarship support", "Scholarship holder"),
    "Displaced": ("Not displaced", "Displaced student"),
    "Educational special needs": ("No educational special needs", "Educational special needs"),
    "International": ("Domestic student", "International student"),
    "Daytime/evening attendance": ("Evening attendance", "Daytime attendance"),
    "Gender": ("Gender code 0", "Gender code 1"),
}

BINARY_OPTION_LABELS = {
    "Daytime/evening attendance": ("Evening", "Daytime"),
    "Gender": ("Code 0", "Code 1"),
}


def binary_options(field: str):
    """Return (label, value) choices for a binary form field."""
    no_label, yes_label = BINARY_OPTION_LABELS.get(field, ("No", "Yes"))
    return [(no_label, 0), (yes_label, 1)]

COUNT_FEATURES = {
    "Age at enrollment",
    "Application order",
    "total_enrolled_units",
    "total_approved_units",
    "total_credited_units",
    "total_evaluations",
    "total_without_evaluations",
    "failed_units_sem1",
    "failed_units_sem2",
    "total_failed_units",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
}


def format_value(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if feature in FRACTION_FEATURES:
        return f"{float(value):.0%}"
    if feature in {"Unemployment rate", "Inflation rate"}:
        return f"{float(value):.2f}%"
    if feature == "GDP":
        return f"{float(value):.2f}"
    if feature in COUNT_FEATURES:
        return str(int(round(float(value))))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value)


def reason_text(feature: str, value, direction: str) -> str:
    """Turn a feature and its SHAP direction into a short, value-aware reason."""
    label = pretty_name(feature)
    increasing = direction == "increases risk"

    if feature in {"overall_approval_rate", "approval_rate_sem1", "approval_rate_sem2"}:
        rate = float(value)
        quality = "Low" if rate < 0.5 else ("Moderate" if rate < 0.8 else "High")
        return f"{quality} {label.lower()} ({format_value(feature, value)})"
    if feature in {"no_approved_sem1", "no_approved_sem2", "no_approved_both_semesters"}:
        return label if int(value) == 1 else f"Approved units present ({label.lower()})"
    if feature in {"low_approval_sem1", "low_approval_sem2", "low_approval_both_semesters"}:
        return label if int(value) == 1 else f"Approval rate is not low ({label.lower()})"
    if feature in {"failed_units_sem1", "failed_units_sem2", "total_failed_units"}:
        return f"{label} ({format_value(feature, value)})"
    if feature in {"effective_avg_grade_sem1", "effective_avg_grade_sem2", "overall_average_grade"}:
        grade = float(value)
        if grade <= 0:
            return f"No graded approved units ({label.lower()})"
        quality = "Low" if grade < 10 else ("Moderate" if grade < 14 else "High")
        return f"{quality} {label.lower()} ({format_value(feature, value)})"
    if feature == "grade_difference_sem2_minus_sem1":
        return f"Semester 2 grade change ({format_value(feature, value)})"
    if feature == "approved_difference_sem2_minus_sem1":
        return f"Semester 2 approved-unit change ({format_value(feature, value)})"
    if feature == "approval_rate_difference_sem2_minus_sem1":
        return f"Semester 2 approval-rate change ({format_value(feature, value)})"
    if feature == "failed_difference_sem2_minus_sem1":
        return f"Semester 2 failed-unit change ({format_value(feature, value)})"
    if feature in BINARY_LABELS:
        return BINARY_LABELS[feature][1 if int(value) == 1 else 0]
    if feature in {"Admission grade", "Previous qualification (grade)"}:
        return f"{label} ({format_value(feature, value)})"
    if feature == "Age at enrollment":
        return f"Age at enrollment ({format_value(feature, value)})"
    if feature in CATEGORICAL_CODE_COLS:
        return f"{label} (code {int(float(value))})"
    if feature in {"Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"}:
        count = int(value)
        semester = "1" if "1st" in feature else "2"
        return f"{count} approved units in semester {semester}"
    if feature in {"Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)"}:
        count = int(value)
        semester = "1" if "1st" in feature else "2"
        return f"{count} enrolled units in semester {semester}"

    suffix = " - increases risk" if increasing else " - reduces risk"
    return f"{label} = {format_value(feature, value)}{suffix}"
