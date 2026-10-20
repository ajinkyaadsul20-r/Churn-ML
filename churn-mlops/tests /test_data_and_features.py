import numpy as np

from churn_ml.data import NUMERIC, RAW_FEATURES, TARGET, generate_churn_data
from churn_ml.features import ENGINEERED, add_features, build_preprocessor


def test_generator_is_reproducible():
    a = generate_churn_data(500, seed=1)
    b = generate_churn_data(500, seed=1)
    assert a.equals(b)


def test_generator_schema_and_rate():
    df = generate_churn_data(3000, seed=0)
    assert set(RAW_FEATURES + [TARGET]).issubset(df.columns)
    assert 0.1 < df[TARGET].mean() < 0.5
    assert df["monthly_charges"].isna().any()  # missingness injected


def test_engineered_features():
    df = generate_churn_data(200, seed=0)
    out = add_features(df)
    assert set(ENGINEERED).issubset(out.columns)
    row = out.dropna(subset=["monthly_charges"]).iloc[0]
    assert np.isclose(row["total_charges"], row["tenure_months"] * row["monthly_charges"])


def test_preprocessor_handles_nan_and_unseen_category():
    df = generate_churn_data(300, seed=0)
    X = df[RAW_FEATURES]
    pre = build_preprocessor().fit(X)
    X_new = X.head(5).copy()
    X_new.loc[X_new.index[0], "contract"] = "Brand new contract"
    X_new.loc[X_new.index[1], "usage_gb"] = np.nan
    out = pre.transform(X_new)
    assert not np.isnan(np.asarray(out.todense() if hasattr(out, "todense") else out)).any()
    assert len(NUMERIC) > 0
