"""Standardized evaluation metrics for universal classification tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute standard classification metrics on held-out test predictions.

    Args:
        y_true: 1D array of true integer labels.
        y_pred: 1D array of predicted integer labels.
        y_prob: 2D array of predicted class probabilities (N, C) or None.

    Returns:
        Dictionary of metric scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(y_true)
    n_classes = len(classes)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # MCC is robust to class imbalance (relevant here: some tasks have <1%
        # positive rate) and stays well-defined even in degenerate edge cases
        # where F1/accuracy can be misleadingly high.
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Macro ROC-AUC
    if y_prob is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                # Handle binary case: if prob is (N, 2), slice column 1; if (N,), use as is
                prob_1 = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_1))
            else:
                # Multi-class OvR Macro AUROC
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.debug(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Macro AUPRC (average precision) - much more informative than ROC-AUC
    # under severe class imbalance, which several tasks in this benchmark have.
    if y_prob is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                prob_1 = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
                metrics["auprc"] = float(average_precision_score(y_true, prob_1))
            else:
                y_true_onehot = np.zeros((len(y_true), n_classes))
                class_to_idx = {c: i for i, c in enumerate(classes)}
                for i, c in enumerate(y_true):
                    y_true_onehot[i, class_to_idx[c]] = 1
                metrics["auprc"] = float(
                    average_precision_score(y_true_onehot, y_prob, average="macro")
                )
        except Exception as e:
            logger.debug(f"Could not compute AUPRC: {e}")
            metrics["auprc"] = float("nan")
    else:
        metrics["auprc"] = float("nan")

    # Log Loss
    if y_prob is not None and n_classes >= 2:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=classes))
        except Exception:
            metrics["log_loss"] = float("nan")
    else:
        metrics["log_loss"] = float("nan")

    return metrics
