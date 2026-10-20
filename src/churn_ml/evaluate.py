"""Metrics, cost-sensitive thresholding and report plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_cost(y_true, y_prob, threshold: float, c_fn: float, c_fp: float) -> float:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * c_fn + fp * c_fp)


def optimal_threshold(y_true, y_prob, c_fn: float, c_fp: float) -> tuple[float, list[tuple[float, float]]]:
    """Pick the probability threshold that minimises expected business cost."""
    grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
    curve = [(float(t), expected_cost(y_true, y_prob, t, c_fn, c_fp)) for t in grid]
    best = min(curve, key=lambda x: x[1])[0]
    return best, curve


def compute_metrics(y_true, y_prob, threshold: float, c_fn: float, c_fp: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    model_cost = expected_cost(y_true, y_prob, threshold, c_fn, c_fp)
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "expected_cost": model_cost,
        "baseline_cost_no_action": float(n_pos * c_fn),
        "baseline_cost_contact_all": float(n_neg * c_fp),
        "cost_saving_vs_best_baseline_pct": float(
            100 * (1 - model_cost / max(min(n_pos * c_fn, n_neg * c_fp), 1))
        ),
    }


def save_reports(pipeline, X_test, y_test, y_prob, threshold, cost_curve, out_dir: Path, seed: int) -> dict:
    """Write diagnostic plots and return permutation importances."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=140)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (y_prob >= threshold).astype(int), ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    ts, cs = zip(*cost_curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, cs)
    ax.axvline(threshold, color="red", ls="--", label=f"optimal = {threshold:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost (OOF train)", title="Cost vs threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cost_curve.png", dpi=140)
    plt.close(fig)

    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1
    )
    order = np.argsort(result.importances_mean)
    names = np.array(X_test.columns)[order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, result.importances_mean[order], xerr=result.importances_std[order])
    ax.set(xlabel="Drop in ROC-AUC when shuffled", title="Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=140)
    plt.close(fig)

    return {n: float(v) for n, v in zip(X_test.columns, result.importances_mean)}
