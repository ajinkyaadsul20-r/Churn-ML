"""Inference wrapper shared by the CLI and the REST API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import RAW_FEATURES


def risk_tier(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


class ChurnPredictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        art = Path(artifacts_dir)
        self.model = joblib.load(art / "model.joblib")
        self.metadata: dict[str, Any] = json.loads((art / "metadata.json").read_text())
        self.threshold: float = self.metadata["threshold"]

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        df = pd.DataFrame(records).reindex(columns=RAW_FEATURES)
        probs = self.model.predict_proba(df)[:, 1]
        return [
            {
                "churn_probability": round(float(p), 4),
                "will_churn": bool(p >= self.threshold),
                "risk_tier": risk_tier(float(p)),
            }
            for p in probs
        ]
