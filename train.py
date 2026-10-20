"""Training pipeline: CV model selection -> calibration -> cost-optimal
threshold (out-of-fold, no test leakage) -> held-out evaluation -> artifacts."""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from . import __version__
from .data import RAW_FEATURES, TARGET, dataframe_hash, load_or_generate
from .evaluate import compute_metrics, optimal_threshold, save_reports
from .models import build_candidates

log = logging.getLogger("churn_ml.train")


def run_training(cfg: dict[str, Any], artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    seed = cfg["seed"]
    art = Path(artifacts_dir or cfg["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    biz = cfg["business"]
    tcfg = cfg["training"]

    # ---- data ----------------------------------------------------------
    df = load_or_generate(cfg["data"]["raw_path"], cfg["data"]["n_samples"], seed)
    X, y = df[RAW_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed
    )
    log.info("Data: %d rows | churn rate %.1f%%", len(df), 100 * y.mean())

    # ---- model selection via stratified CV ---------------------------------
    cv = StratifiedKFold(tcfg["cv_folds"], shuffle=True, random_state=seed)
    candidates = build_candidates(cfg, seed)
    cv_results: dict[str, dict[str, float]] = {}
    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, n_jobs=1,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        )
        cv_results[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "pr_auc_mean": float(scores["test_pr_auc"].mean()),
        }
        log.info("CV %-24s ROC-AUC %.4f +/- %.4f", name,
                 cv_results[name]["roc_auc_mean"], cv_results[name]["roc_auc_std"])
    best_name = max(cv_results, key=lambda k: cv_results[k][f"{tcfg['selection_metric']}_mean"])
    log.info("Selected model: %s", best_name)

    # ---- calibrated final model --------------------------------------------
    final = CalibratedClassifierCV(
        clone(candidates[best_name]), method=tcfg["calibration"], cv=3
    )
    # Out-of-fold probabilities on TRAIN only -> choose threshold without peeking at test.
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, cost_curve = optimal_threshold(
        y_train.to_numpy(), oof, biz["cost_false_negative"], biz["cost_false_positive"]
    )
    final.fit(X_train, y_train)

    # ---- held-out evaluation -------------------------------------------------
    y_prob = final.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(
        y_test.to_numpy(), y_prob, threshold, biz["cost_false_negative"], biz["cost_false_positive"]
    )
    importances = save_reports(final, X_test, y_test, y_prob, threshold, cost_curve, art / "reports", seed)
    log.info("Test ROC-AUC %.4f | PR-AUC %.4f | recall %.3f | precision %.3f | cost saving vs best baseline %.1f%%",
             metrics["roc_auc"], metrics["pr_auc"], metrics["recall"], metrics["precision"],
             metrics["cost_saving_vs_best_baseline_pct"])

    # ---- persist ---------------------------------------------------------------
    joblib.dump(final, art / "model.joblib")
    metadata = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_family": best_name,
        "calibration": tcfg["calibration"],
        "threshold": threshold,
        "features": RAW_FEATURES,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data_hash": dataframe_hash(df),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cv_results": cv_results,
        "permutation_importance": importances,
    }
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (art / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"metrics": metrics, "metadata": metadata}
