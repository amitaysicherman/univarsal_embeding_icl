#!/usr/bin/env python3
"""Aggregate and display benchmark results across all 7 domains."""

import glob
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results" / "baselines"
files = sorted(list(results_dir.glob("baselines_*.json")))

print("=" * 96)
print(f"{'Domain':<14} | {'Model':<18} | {'Mean Acc':<10} | {'Mean AUROC':<12} | {'Mean F1':<10} | {'Evaluations'}")
print("-" * 96)

overall = {
    "linear": {"acc": [], "auc": [], "f1": []},
    "knn": {"acc": [], "auc": [], "f1": []},
    "mlp": {"acc": [], "auc": [], "f1": []},
    "tabicl_zeroshot": {"acc": [], "auc": [], "f1": []},
}

for fpath in files:
    dom = fpath.stem.replace("baselines_", "")
    with open(fpath) as f:
        data = json.load(f)

    dom_stats = {
        "linear": {"acc": [], "auc": [], "f1": []},
        "knn": {"acc": [], "auc": [], "f1": []},
        "mlp": {"acc": [], "auc": [], "f1": []},
        "tabicl_zeroshot": {"acc": [], "auc": [], "f1": []},
    }

    for key, models in data.items():
        for m_name in ["linear", "knn", "mlp", "tabicl_zeroshot"]:
            if m_name in models and "accuracy" in models[m_name]:
                res = models[m_name]
                dom_stats[m_name]["acc"].append(res["accuracy"])
                if not np.isnan(res["roc_auc"]):
                    dom_stats[m_name]["auc"].append(res["roc_auc"])
                dom_stats[m_name]["f1"].append(res["macro_f1"])

                overall[m_name]["acc"].append(res["accuracy"])
                if not np.isnan(res["roc_auc"]):
                    overall[m_name]["auc"].append(res["roc_auc"])
                overall[m_name]["f1"].append(res["macro_f1"])

    for m_name in ["linear", "knn", "mlp", "tabicl_zeroshot"]:
        acc = np.mean(dom_stats[m_name]["acc"]) * 100 if dom_stats[m_name]["acc"] else 0.0
        auc = np.mean(dom_stats[m_name]["auc"]) if dom_stats[m_name]["auc"] else 0.0
        f1 = np.mean(dom_stats[m_name]["f1"]) * 100 if dom_stats[m_name]["f1"] else 0.0
        n_ev = len(dom_stats[m_name]["acc"])
        print(f"{dom:<14} | {m_name:<18} | {acc:6.2f}%    | {auc:.4f}       | {f1:6.2f}%    | {n_ev}")
    print("-" * 96)

print("=" * 96)
print(f"{'OVERALL':<14} | {'-'*18} | {'-'*10} | {'-'*12} | {'-'*10} |")
for m_name in ["linear", "knn", "mlp", "tabicl_zeroshot"]:
    acc = np.mean(overall[m_name]["acc"]) * 100
    auc = np.mean(overall[m_name]["auc"])
    f1 = np.mean(overall[m_name]["f1"]) * 100
    n_ev = len(overall[m_name]["acc"])
    print(f"{'ALL DOMAINS':<14} | {m_name:<18} | {acc:6.2f}%    | {auc:.4f}       | {f1:6.2f}%    | {n_ev}")
print("=" * 96)
