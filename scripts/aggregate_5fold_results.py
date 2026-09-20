#!/usr/bin/env python3
"""Aggregate 5-Fold Cross-Generalization evaluation results into a master report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("aggregate_5fold")


def main():
    parser = argparse.ArgumentParser(description="Aggregate 5-Fold Cross-Generalization Results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--output-file", type=str, default="results/cross_generalization_5fold_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    regimes = ["seen", "unseen_tasks", "unseen_encoders", "unseen_both"]
    
    # Store all evaluation pairs pooled across folds
    pooled_pairs: Dict[str, Dict[str, Any]] = {r: {} for r in regimes}
    fold_summaries: List[Dict[str, Any]] = []

    for fold in range(args.num_folds):
        fold_file = results_dir / f"fold_{fold}" / "universal_generalization_summary.json"
        if not fold_file.exists():
            logger.warning(f"Fold {fold} summary not found at {fold_file}")
            continue

        with open(fold_file, "r") as f:
            data = json.load(f)
            fold_summaries.append(data)

        for regime in regimes:
            if regime in data:
                res = data[regime].get("results", {})
                for key, metrics in res.items():
                    # If already present (e.g. seen), average or track by fold
                    if key not in pooled_pairs[regime]:
                        pooled_pairs[regime][key] = []
                    pooled_pairs[regime][key].append(metrics)

    if not fold_summaries:
        logger.error("No valid fold summaries found to aggregate.")
        return

    logger.info(f"Loaded {len(fold_summaries)} valid folds.")

    master_summary = {
        "num_folds_evaluated": len(fold_summaries),
        "regimes": {},
        "per_domain": {},
    }

    print("\n" + "=" * 80)
    print("      UNIVERSAL TABICL: 5-FOLD CROSS-GENERALIZATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Generalization Regime':<20} | {'Total Pairs':<12} | {'Mean Acc (%)':<15} | {'Mean AUROC':<15} | {'Mean F1 (%)':<15}")
    print("-" * 80)

    for regime in regimes:
        pairs_dict = pooled_pairs[regime]
        if not pairs_dict:
            continue

        # For each pair, take average metrics across folds
        acc_list, auroc_list, f1_list = [], [], []
        domain_breakdown: Dict[str, Dict[str, List[float]]] = {}

        for key, metrics_list in pairs_dict.items():
            domain, task_id, enc_id = key.split("::")
            avg_acc = np.mean([m["accuracy"] for m in metrics_list])
            avg_auroc = np.mean([m["roc_auc"] for m in metrics_list if not np.isnan(m["roc_auc"])])
            avg_f1 = np.mean([m["macro_f1"] for m in metrics_list])

            acc_list.append(avg_acc)
            if not np.isnan(avg_auroc):
                auroc_list.append(avg_auroc)
            f1_list.append(avg_f1)

            if domain not in domain_breakdown:
                domain_breakdown[domain] = {"acc": [], "auroc": [], "f1": []}
            domain_breakdown[domain]["acc"].append(avg_acc)
            if not np.isnan(avg_auroc):
                domain_breakdown[domain]["auroc"].append(avg_auroc)
            domain_breakdown[domain]["f1"].append(avg_f1)

        mean_acc = float(np.mean(acc_list)) * 100
        std_acc = float(np.std(acc_list)) * 100
        mean_auc = float(np.mean(auroc_list))
        std_auc = float(np.std(auroc_list))
        mean_f1 = float(np.mean(f1_list)) * 100
        std_f1 = float(np.std(f1_list)) * 100

        master_summary["regimes"][regime] = {
            "total_pairs_evaluated": len(pairs_dict),
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_auroc": mean_auc,
            "std_auroc": std_auc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "domain_breakdown": {
                d: {
                    "accuracy": float(np.mean(vals["acc"])) * 100,
                    "auroc": float(np.mean(vals["auroc"])) if vals["auroc"] else 0.0,
                    "f1": float(np.mean(vals["f1"])) * 100,
                    "count": len(vals["acc"]),
                }
                for d, vals in domain_breakdown.items()
            },
        }

        print(
            f"{regime:<20} | {len(pairs_dict):<12} | "
            f"{mean_acc:>6.2f} ± {std_acc:<6.2f} | "
            f"{mean_auc:>6.4f} ± {std_auc:<6.4f} | "
            f"{mean_f1:>6.2f} ± {std_f1:<6.2f}"
        )

    print("=" * 80 + "\n")

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(master_summary, f, indent=2)
    logger.info(f"Saved master 5-fold cross-generalization summary to {out_file}")


if __name__ == "__main__":
    main()
