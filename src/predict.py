"""Load saved artifacts, predict dropout risk, and explain the result."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .feature_engineering import engineer_features, pretty_name, reason_text
from .preprocessing import transform_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

DISCLAIMER = (
    "This prediction is a decision-support tool and should not be used as the "
    "final decision. Human review is required."
)

RECOMMENDATIONS = {
    "Academic": [
        "Arrange tutoring or extra academic support.",
        "Review previous qualification readiness.",
        "Contact course teachers for academic feedback.",
        "Create a grade improvement plan.",
    ],
    "Engagement": [
        "Contact the student for a supportive check-in.",
        "Create a study recovery plan.",
        "Monitor assignment and exam participation.",
        "Assign a peer buddy or mentor.",
        "Schedule a counselor meeting.",
    ],
    "Financial": [
        "Check scholarship eligibility.",
        "Contact the financial aid office.",
        "Offer payment plan information.",
        "Verify tuition fee status.",
    ],
    "Demographic": [
        "Offer student support services.",
        "Connect the student with counseling.",
        "Provide peer support or mentorship.",
        "Check whether the student needs relocation or special assistance.",
    ],
    "Institutional/Course": [
        "Contact the course coordinator.",
        "Review course difficulty or workload.",
        "Check whether the student is in the correct course.",
        "Offer academic advising.",
    ],
    "Macro-Economic": [
        "Provide institutional support resources.",
        "Monitor the student more closely.",
        "Offer career and financial counseling if available.",
    ],
}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class DropoutPredictor:
    """Reusable predictor for one student or a Streamlit form."""

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.model = joblib.load(self.models_dir / "dropout_model.joblib")
        self.preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")
        self.feature_columns = _load_json(self.models_dir / "feature_columns.json")
        threshold_info = _load_json(self.models_dir / "threshold.json")
        self.threshold = float(threshold_info["best_threshold"])
        self.risk_bands = threshold_info.get(
            "risk_bands",
            {
                "Low": [0.00, 0.35],
                "Medium": [0.35, 0.65],
                "High": [0.65, 0.85],
                "Critical": [0.85, 1.01],
            },
        )
        self.category_map = _load_json(self.models_dir / "category_map.json")
        self.default_input_values = _load_json(
            self.models_dir / "default_input_values.json"
        )
        self.raw_features = list(self.default_input_values.keys())
        self.explainer = shap.TreeExplainer(self.model)

    def risk_level(self, probability: float) -> str:
        for level, band in self.risk_bands.items():
            lower, upper = band
            if lower <= probability < upper:
                return level
        return "Critical"

    def _shap_values(self, X_row: pd.DataFrame) -> np.ndarray:
        values = self.explainer.shap_values(X_row)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values.reshape(-1)

    def explain_row(
        self,
        raw_row,
        feature_row,
        X_row,
        top_n=5,
        risk_level=None,
        allowed_features=None,
    ):
        shap_values = self._shap_values(X_row)
        if shap_values.shape[0] != len(self.feature_columns):
            raise ValueError("SHAP output does not match the saved feature columns.")

        allowed = set(allowed_features) if allowed_features is not None else None
        records = []

        for index, feature in enumerate(self.feature_columns):
            if allowed is not None and feature not in allowed:
                continue
            if feature in raw_row.columns:
                value = raw_row.iloc[0][feature]
            else:
                value = feature_row.iloc[0].get(feature, np.nan)
            if hasattr(value, "item"):
                value = value.item()

            shap_value = float(shap_values[index])
            direction = "increases risk" if shap_value > 0 else "decreases risk"
            records.append(
                {
                    "feature": feature,
                    "feature_label": pretty_name(feature),
                    "category": self.category_map.get(feature, "Engagement"),
                    "value": value,
                    "encoded_value": round(float(X_row.iloc[0][feature]), 6),
                    "shap_value": round(shap_value, 6),
                    "direction": direction,
                    "reason": reason_text(feature, value, direction),
                }
            )

        if not records:
            return [], []

        total = sum(abs(row["shap_value"]) for row in records) or 1.0
        for row in records:
            row["percent_contribution"] = round(
                100 * abs(row["shap_value"]) / total, 2
            )

        positive = sorted(
            [row for row in records if row["shap_value"] > 0],
            key=lambda row: row["shap_value"],
            reverse=True,
        )
        negative = sorted(
            [row for row in records if row["shap_value"] <= 0],
            key=lambda row: abs(row["shap_value"]),
            reverse=True,
        )
        if risk_level == "Low":
            # For a low-risk student, show the strongest factors overall,
            # including protective factors, instead of only risk-increasing ones.
            top_reasons = sorted(
                records, key=lambda row: abs(row["shap_value"]), reverse=True
            )[:top_n]
        else:
            top_reasons = (positive + negative)[:top_n]

        category_totals = {}
        for row in records:
            category = row["category"]
            category_totals[category] = (
                category_totals.get(category, 0.0) + abs(row["shap_value"])
            )
        category_contributions = [
            {
                "category": category,
                "percent_contribution": round(100 * value / total, 2),
            }
            for category, value in sorted(
                category_totals.items(), key=lambda item: item[1], reverse=True
            )
        ]
        return top_reasons, category_contributions

    def recommend(self, category_contributions, top_reasons, risk_level):
        if risk_level == "Low":
            return [
                "No immediate intervention is indicated.",
                "Continue routine monitoring and encourage current academic habits.",
                "Share available academic, financial, and wellbeing support resources.",
            ]

        top_category = (
            category_contributions[0]["category"]
            if category_contributions
            else "Engagement"
        )
        recommendations = list(
            RECOMMENDATIONS.get(top_category, RECOMMENDATIONS["Engagement"])
        )
        features = {row["feature"] for row in top_reasons}

        if features & {"Debtor", "Tuition fees up to date"}:
            recommendations.append(
                "Review financial aid, scholarship, and payment-plan options."
            )
        if features & {
            "overall_approval_rate",
            "total_failed_units",
            "failed_units_sem1",
            "failed_units_sem2",
        }:
            recommendations.append(
                "Create a semester recovery plan with the academic advisor."
            )
        if risk_level in {"High", "Critical"}:
            recommendations.append(
                "Schedule a supportive check-in with the student soon."
            )

        unique = []
        for item in recommendations:
            if item not in unique:
                unique.append(item)
        return unique[:6]

    def predict(self, input_data: dict, explanation_features=None) -> dict:
        """Predict one student and return a JSON-compatible explanation.

        Pass ``explanation_features`` to limit SHAP reasons to the fields that
        were actually collected by a simplified form.
        """
        input_data = dict(input_data or {})
        unknown_fields = [
            field for field in input_data if field not in self.raw_features
        ]
        missing_fields = []
        values = {}

        for feature in self.raw_features:
            value = input_data.get(feature)
            if _is_missing(value):
                value = self.default_input_values[feature]
                missing_fields.append(feature)
            values[feature] = value

        raw_row = pd.DataFrame([values], columns=self.raw_features)
        raw_row = raw_row.replace(
            {"Yes": 1, "No": 0, "yes": 1, "no": 0, "TRUE": 1, "FALSE": 0}
        )
        for column in self.raw_features:
            raw_row[column] = pd.to_numeric(raw_row[column], errors="coerce")
            if raw_row[column].isna().any():
                raw_row[column] = raw_row[column].fillna(
                    self.default_input_values[column]
                )

        feature_row = engineer_features(raw_row)
        X_row = transform_features(feature_row, self.preprocessor)
        X_row = X_row.reindex(columns=self.feature_columns)

        probability = float(self.model.predict_proba(X_row)[0, 1])
        risk = self.risk_level(probability)
        top_reasons, category_contributions = self.explain_row(
            raw_row,
            feature_row,
            X_row,
            risk_level=risk,
            allowed_features=explanation_features,
        )
        recommendations = self.recommend(
            category_contributions, top_reasons, risk
        )

        return {
            "dropout_probability": round(probability, 4),
            "dropout_probability_percent": round(probability * 100, 1),
            "risk_level": risk,
            "predicted_dropout": int(probability >= self.threshold),
            "threshold_used": round(self.threshold, 4),
            "top_reasons": top_reasons,
            "category_contributions": category_contributions,
            "recommendations": recommendations,
            "missing_fields_filled": missing_fields,
            "ignored_fields": unknown_fields,
            "disclaimer": DISCLAIMER,
        }


_DEFAULT_PREDICTOR = None


def get_predictor() -> DropoutPredictor:
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = DropoutPredictor()
    return _DEFAULT_PREDICTOR


def predict_dropout(input_data: dict, explanation_features=None) -> dict:
    """Convenience function used by the chatbot and notebooks."""
    return get_predictor().predict(
        input_data, explanation_features=explanation_features
    )
