"""FastAPI service.  Run:  uvicorn churn_ml.api:app --port 8000"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import ChurnPredictor

state: dict[str, ChurnPredictor] = {}


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, examples=[8])
    monthly_charges: float | None = Field(None, ge=0, examples=[89.5])
    num_products: int = Field(..., ge=1, le=10, examples=[1])
    support_calls: int = Field(..., ge=0, examples=[4])
    late_payments: int = Field(..., ge=0, examples=[2])
    senior_citizen: Literal[0, 1] = 0
    usage_gb: float | None = Field(None, ge=0, examples=[35.0])
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["Fiber", "DSL", "None"]
    payment_method: Literal["Electronic check", "Credit card", "Bank transfer", "Mailed check"]


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_tier: Literal["low", "medium", "high"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["predictor"] = ChurnPredictor(os.getenv("MODEL_DIR", "artifacts"))
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def _predictor() -> ChurnPredictor:
    if "predictor" not in state:
        raise HTTPException(503, "Model not loaded")
    return state["predictor"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model-info")
def model_info() -> dict:
    md = _predictor().metadata
    keys = ("model_family", "trained_at", "threshold", "data_hash", "sklearn", "package_version")
    return {k: md[k] for k in keys}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> dict:
    return _predictor().predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[dict]:
    if not customers or len(customers) > 1000:
        raise HTTPException(422, "Batch size must be between 1 and 1000")
    return _predictor().predict([c.model_dump() for c in customers])
