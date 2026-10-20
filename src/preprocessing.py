"""Leakage-safe preprocessing: frequency encoding and median imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_preprocessor(
    X_train: pd.DataFrame,
    categorical_features,
    numeric_features,
    feature_columns,
) -> dict:
    """Learn category frequencies and numeric medians from the training set."""
    frequency_maps = {}
    for column in categorical_features:
        values = X_train[column].astype("string").fillna("missing")
        frequencies = values.value_counts(normalize=True)
        frequency_maps[column] = {
            str(key): float(value) for key, value in frequencies.items()
        }

    medians = {}
    for column in numeric_features:
        values = pd.to_numeric(X_train[column], errors="coerce")
        median = values.median()
        medians[column] = float(median) if pd.notna(median) else 0.0

    return {
        "categorical_features": list(categorical_features),
        "numeric_features": list(numeric_features),
        "frequency_maps": frequency_maps,
        "medians": medians,
        "feature_columns": list(feature_columns),
    }


def transform_features(data: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """Apply a fitted preprocessor to any split or to one new student."""
    output = pd.DataFrame(index=data.index)

    for column in preprocessor["categorical_features"]:
        values = data[column].astype("string").fillna("missing")
        mapping = preprocessor["frequency_maps"][column]
        output[column] = values.map(mapping).astype(float).fillna(0.0)

    for column in preprocessor["numeric_features"]:
        values = pd.to_numeric(data[column], errors="coerce")
        output[column] = values.fillna(preprocessor["medians"][column])

    output = output.reindex(columns=preprocessor["feature_columns"])
    output = output.replace([np.inf, -np.inf], np.nan)
    return output.fillna(0.0).astype(float)
