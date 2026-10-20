"""Feature engineering + preprocessing (all inside the sklearn Pipeline,
so training and serving can never drift apart)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

ENGINEERED = ["total_charges", "charges_per_product", "high_support_flag"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive business features from raw columns (NaN-safe)."""
    out = df.copy()
    out["total_charges"] = out["tenure_months"] * out["monthly_charges"]
    out["charges_per_product"] = out["monthly_charges"] / out["num_products"].replace(0, np.nan)
    out["high_support_flag"] = (out["support_calls"] >= 3).astype(int)
    return out


def build_preprocessor() -> Pipeline:
    numeric_cols = NUMERIC + ENGINEERED
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    column_transformer = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, CATEGORICAL)]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features, validate=False)),
            ("columns", column_transformer),
        ]
    )
