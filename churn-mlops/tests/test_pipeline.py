import copy
import json

from churn_ml.config import load_config
from churn_ml.predict import ChurnPredictor
from churn_ml.train import run_training


def _fast_cfg(tmp_path):
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["data"]["n_samples"] = 2500
    cfg["data"]["raw_path"] = str(tmp_path / "churn.csv")
    cfg["training"]["cv_folds"] = 3
    cfg["training"]["models"]["random_forest"]["n_estimators"] = 25
    cfg["training"]["models"]["hist_gradient_boosting"]["max_iter"] = 40
    return cfg


def test_train_then_predict_roundtrip(tmp_path):
    art = tmp_path / "artifacts"
    res = run_training(_fast_cfg(tmp_path), art)

    assert res["metrics"]["roc_auc"] > 0.7          # real signal learned
    assert 0.0 < res["metrics"]["threshold"] < 1.0
    for f in ("model.joblib", "metrics.json", "metadata.json"):
        assert (art / f).exists()
    assert (art / "reports" / "roc_pr.png").exists()
    assert json.loads((art / "metadata.json").read_text())["data_hash"]

    predictor = ChurnPredictor(art)
    risky = {
        "tenure_months": 3, "monthly_charges": 105.0, "num_products": 1, "support_calls": 5,
        "late_payments": 3, "senior_citizen": 0, "usage_gb": 10.0,
        "contract": "Month-to-month", "internet_service": "Fiber", "payment_method": "Electronic check",
    }
    loyal = {
        "tenure_months": 60, "monthly_charges": 55.0, "num_products": 4, "support_calls": 0,
        "late_payments": 0, "senior_citizen": 0, "usage_gb": 150.0,
        "contract": "Two year", "internet_service": "DSL", "payment_method": "Credit card",
    }
    r_risky, r_loyal = predictor.predict([risky, loyal])
    assert r_risky["churn_probability"] > r_loyal["churn_probability"]
    assert r_risky["risk_tier"] in {"medium", "high"}
    assert r_loyal["risk_tier"] == "low"
