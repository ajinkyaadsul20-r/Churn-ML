import copy

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from churn_ml.config import load_config  # noqa: E402
from churn_ml.train import run_training  # noqa: E402

PAYLOAD = {
    "tenure_months": 5, "monthly_charges": 95.0, "num_products": 1, "support_calls": 4,
    "late_payments": 2, "senior_citizen": 0, "usage_gb": 12.0,
    "contract": "Month-to-month", "internet_service": "Fiber", "payment_method": "Electronic check",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["data"].update(n_samples=2000, raw_path=str(tmp_path / "d.csv"))
    cfg["training"]["cv_folds"] = 3
    cfg["training"]["models"]["random_forest"]["n_estimators"] = 20
    cfg["training"]["models"]["hist_gradient_boosting"]["max_iter"] = 30
    run_training(cfg, tmp_path / "art")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "art"))
    from churn_ml.api import app

    with TestClient(app) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert "threshold" in client.get("/model-info").json()


def test_predict_single_and_batch(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert len(client.post("/predict/batch", json=[PAYLOAD, PAYLOAD]).json()) == 2


def test_validation_rejects_bad_input(client):
    bad = {**PAYLOAD, "contract": "Lifetime"}
    assert client.post("/predict", json=bad).status_code == 422
