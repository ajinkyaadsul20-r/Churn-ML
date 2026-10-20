"""Reproducible synthetic telecom-churn data generator.

The generator embeds realistic, non-linear signal (contract type, tenure,
support calls, payment friction...) plus missing values, so the whole
pipeline is exercised without needing any external download.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "churn"
ID_COLUMN = "customer_id"

CATEGORICAL = ["contract", "internet_service", "payment_method"]
NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "num_products",
    "support_calls",
    "late_payments",
    "senior_citizen",
    "usage_gb",
]
RAW_FEATURES = NUMERIC + CATEGORICAL

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber", "DSL", "None"]
PAYMENT = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]


def generate_churn_data(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer table with a binary ``churn`` label."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n_samples)
    contract = rng.choice(CONTRACTS, n_samples, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n_samples, p=[0.45, 0.35, 0.20])
    payment = rng.choice(PAYMENT, n_samples, p=[0.35, 0.25, 0.25, 0.15])
    num_products = rng.integers(1, 6, n_samples)
    support_calls = rng.poisson(1.5, n_samples)
    late_payments = rng.poisson(0.8, n_samples)
    senior = rng.binomial(1, 0.16, n_samples)
    usage_gb = rng.gamma(2.0, 40.0, n_samples)

    base_price = pd.Series(internet).map({"Fiber": 70.0, "DSL": 50.0, "None": 20.0}).to_numpy()
    monthly = np.clip(base_price + 5.0 * num_products + rng.normal(0, 8, n_samples), 15, None)

    logit = (
        -1.3
        - 0.045 * tenure
        + 1.15 * (contract == "Month-to-month")
        - 0.95 * (contract == "Two year")
        + 0.55 * (internet == "Fiber")
        + 0.45 * (payment == "Electronic check")
        + 0.35 * np.minimum(support_calls, 6)
        + 0.30 * late_payments
        + 0.014 * (monthly - 65)
        + 0.30 * senior
        - 0.15 * num_products
        - 0.004 * usage_gb
        + 0.25 * ((tenure < 12) & (contract == "Month-to-month"))  # interaction
        + 0.90 * ((support_calls >= 4) & (contract == "Month-to-month"))  # frustrated, uncommitted
        + 0.80 * (late_payments >= 3)  # threshold effect
        + 0.80 * ((internet == "Fiber") & (monthly > 90))  # price-sensitive fiber users
        + 0.55 * (usage_gb < 20)  # disengaged
        - 0.60 * (usage_gb > 120)  # power users stick around
        + rng.normal(0, 0.4, n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"C{100000 + i}" for i in range(n_samples)],
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "num_products": num_products,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "senior_citizen": senior,
            "usage_gb": usage_gb.round(2),
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            TARGET: churn,
        }
    )

    # Inject ~3% missingness in two numeric columns to exercise imputation.
    for col in ("monthly_charges", "usage_gb"):
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan
    return df


def load_or_generate(path: str | Path, n_samples: int, seed: int) -> pd.DataFrame:
    """Load the CSV at ``path``; generate and cache it if it does not exist."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_churn_data(n_samples, seed)
    df.to_csv(path, index=False)
    return df


def dataframe_hash(df: pd.DataFrame) -> str:
    """Stable content hash used for data lineage in model metadata."""
    hashed = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
