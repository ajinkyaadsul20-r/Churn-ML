"""Candidate model zoo, built from config."""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_candidates(cfg: dict[str, Any], seed: int) -> dict[str, Pipeline]:
    params = cfg["training"]["models"]
    estimators: dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(random_state=seed, **params["logistic_regression"]),
        "random_forest": RandomForestClassifier(random_state=seed, **params["random_forest"]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=seed, **params["hist_gradient_boosting"]
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", est)])
        for name, est in estimators.items()
    }
