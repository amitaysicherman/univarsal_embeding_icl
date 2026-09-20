#!/usr/bin/env python3
"""Run standard tabular baselines (Linear, KNN, MLP, TabICL Zero-Shot) on foundation embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from univarsal_embeding.eval.evaluator import compute_classification_metrics
from univarsal_embeding.models.baselines import (
    KNNBaseline,
    LinearBaseline,
    MLPBaseline,
    TabICLZeroShotBaseline,
    XGBoostBaseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_baselines")


def evaluate_task_encoder(
    task_dir: Path,
    encoder_name: str,
    model_types: List[str],
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    emb_path = task_dir / "embeddings" / f"{encoder_name}.npy"
    labels_path = task_dir / "labels.npy"
    folds_path = task_dir / "folds.npy"

    if not emb_path.exists() or not labels_path.exists() or not folds_path.exists():
        logger.warning(f"Missing files in {task_dir} for encoder {encoder_name}")
        return {}

    X = np.load(emb_path)
    y = np.load(labels_path)
    folds = np.load(folds_path)

    # In our strict protocol: 0=test, 1=val, 2=train
    test_idx = np.flatnonzero(folds == 0)
    val_idx = np.flatnonzero(folds == 1)
    train_idx = np.flatnonzero(folds == 2)

    # In case val fold is empty, take a small fraction of train
    if len(val_idx) == 0:
        n_val = max(1, int(len(train_idx) * 0.15))
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Check that test set has at least 2 classes
    if len(np.unique(y_test)) < 2:
        logger.warning(f"Single class in test set for {task_dir.name}, skipping.")
        return {}

    results = {}

    for m_type in model_types:
        t0 = time.time()
        try:
            if m_type == "linear":
                clf = LinearBaseline(C=1.0)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
            elif m_type == "knn":
                clf = KNNBaseline(n_neighbors=5, metric="cosine")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
            elif m_type == "mlp":
                clf = MLPBaseline(hidden_dim=256, device=device, epochs=100)
                clf.fit(X_train, y_train, X_val, y_val)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
            elif m_type == "xgboost":
                clf = XGBoostBaseline()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
            elif m_type == "tabicl_zeroshot":
                tabicl_clf = TabICLZeroShotBaseline(device=device, context_size=512)
                y_prob = tabicl_clf.predict_proba(X_train, y_train, X_test)
                y_pred = np.argmax(y_prob, axis=1)
            else:
                logger.warning(f"Unknown model type: {m_type}")
                continue

            elapsed = time.time() - t0
            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            metrics["elapsed_seconds"] = round(elapsed, 2)
            results[m_type] = metrics
            logger.info(
                f"[{task_dir.name}] [{encoder_name}] [{m_type}] Acc={metrics['accuracy']:.4f} "
                f"AUROC={metrics['roc_auc']:.4f} F1={metrics['macro_f1']:.4f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"Failed {m_type} on {task_dir.name}/{encoder_name}: {e}")
            results[m_type] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline classifiers on foundation embeddings")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--domain", type=str, default=None, help="Optional domain filter")
    parser.add_argument(
        "--models",
        type=str,
        default="xgboost,knn,mlp,tabicl_zeroshot,linear",
        help="Comma-separated model types",
    )
    parser.add_argument("--output-dir", type=str, default="results/baselines")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()

    model_types = [m.strip() for m in args.models.split(",")]
    data_root = Path(args.data_root) / "tasks"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = [args.domain] if args.domain else [d.name for d in data_root.iterdir() if d.is_dir()]
    all_summary = {}

    for d in sorted(domains):
        domain_dir = data_root / d
        if not domain_dir.exists():
            continue
        task_dirs = sorted([p for p in domain_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()])
        if args.limit_tasks:
            task_dirs = task_dirs[: args.limit_tasks]

        logger.info(f"=== Domain: {d} ({len(task_dirs)} tasks) ===")
        domain_results = {}

        for t_dir in task_dirs:
            emb_dir = t_dir / "embeddings"
            if not emb_dir.exists():
                continue
            enc_files = sorted(list(emb_dir.glob("*.npy")))
            for ef in enc_files:
                enc_name = ef.stem
                res = evaluate_task_encoder(t_dir, enc_name, model_types, device=args.device)
                key = f"{t_dir.name}::{enc_name}"
                domain_results[key] = res

        out_file = out_dir / f"baselines_{d}.json"
        with open(out_file, "w") as f:
            json.dump(domain_results, f, indent=2)
        logger.info(f"Saved {d} baseline results to {out_file}")
        all_summary[d] = len(domain_results)

    logger.info(f"Completed baselines evaluation: {all_summary}")


if __name__ == "__main__":
    import torch
    main()
