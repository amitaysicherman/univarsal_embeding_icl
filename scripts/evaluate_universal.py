#!/usr/bin/env python3
"""Evaluate Universal TabICL across the 5 Generalization Regimes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from univarsal_embeding.encoders.registry import DOMAIN_TO_ENCODERS
from univarsal_embeding.eval.evaluator import compute_classification_metrics
from univarsal_embeding.eval.splits import UniversalBenchmarkSplits
from univarsal_embeding.models.tabicl_universal import UniversalTabICL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_universal")


def evaluate_pair(
    model: UniversalTabICL,
    data_root: Path,
    domain: str,
    task_id: str,
    enc_id: str,
    device: str = "cpu",
    context_size: int | None = None,
    num_projections: int = 16,
) -> Dict[str, float]:
    task_dir = data_root / "tasks" / domain / task_id
    emb_path = task_dir / "embeddings" / f"{enc_id}.npy"
    labels_path = task_dir / "labels.npy"
    folds_path = task_dir / "folds.npy"

    if not (emb_path.exists() and labels_path.exists() and folds_path.exists()):
        return {}

    X = np.load(emb_path)
    y = np.load(labels_path)
    folds = np.load(folds_path)

    test_idx = np.flatnonzero(folds == 0)
    train_idx = np.flatnonzero(folds == 2)

    if len(test_idx) == 0 or len(train_idx) == 0 or len(np.unique(y[test_idx])) < 2:
        return {}

    # Context = ALL available training samples (matches official TabICL usage
    # and training-time behavior - no subsampling of context rows). Pass an
    # explicit context_size only as an opt-in cap for tasks too large to fit.
    if context_size is None:
        ctx_idx = train_idx
    else:
        rng = np.random.default_rng(42)
        n_ctx = min(context_size, len(train_idx))
        ctx_idx = rng.choice(train_idx, size=n_ctx, replace=False)

    X_ctx = torch.tensor(X[ctx_idx], dtype=torch.float32, device=device)
    y_ctx_raw = y[ctx_idx]
    classes, inv = np.unique(y_ctx_raw, return_inverse=True)
    label_map = {c: i for i, c in enumerate(classes)}
    y_ctx = torch.tensor(inv, dtype=torch.long, device=device)

    X_test = X[test_idx]
    y_test_raw = y[test_idx]
    y_test_mapped = np.array([label_map.get(lbl, 0) for lbl in y_test_raw])

    query_chunk = 64
    all_probs = []

    with torch.inference_mode():
        for b in range(0, len(X_test), query_chunk):
            q_chunk = torch.tensor(X_test[b : b + query_chunk], dtype=torch.float32, device=device)
            try:
                if num_projections > 1:
                    probs = model.forward_episodic_ensemble(
                        X_ctx, y_ctx, q_chunk, num_projections=num_projections, base_seed=42
                    ).cpu().numpy()
                else:
                    logits = model.forward_episodic(X_ctx, y_ctx, q_chunk, dynamic_rp=True, projection_seed=42)
                    # Softmax in float32: normalizing bf16 logits leaves row
                    # sums off by up to ~0.2%, which sklearn's multi-class
                    # AUROC validator rejects outright.
                    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            except Exception as e:
                logger.warning(f"Inference chunk failed: {e}")
                probs = np.ones((len(q_chunk), len(classes))) / len(classes)
            all_probs.append(probs)

    y_prob = np.vstack(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_classification_metrics(y_test_mapped, y_pred, y_prob)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Universal TabICL across 5 Generalization Regimes")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained LoRA checkpoint path")
    parser.add_argument(
        "--regimes",
        type=str,
        default="seen,unseen_tasks,unseen_encoders,unseen_both,unseen_domain",
        help="Comma-separated list of regimes to evaluate",
    )
    parser.add_argument("--holdout-domain", type=str, default="proteins", help="Domain to hold out for unseen_domain regime")
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="results/universal_eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Fold index (0-4) for 5-fold cross-generalization")
    parser.add_argument("--num-folds", type=int, default=5, help="Total number of folds for cross-generalization")
    parser.add_argument(
        "--num-projections",
        type=int,
        default=16,
        help="Number of random orthogonal projections to ensemble for inference (1 = single projection)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = [r.strip() for r in args.regimes.split(",")]

    # Build domain to tasks map
    domain_to_tasks = {}
    for d_path in (data_root / "tasks").iterdir():
        if d_path.is_dir():
            domain_to_tasks[d_path.name] = sorted(
                [p.name for p in d_path.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
            )

    splits_manager = UniversalBenchmarkSplits(
        domain_to_tasks=domain_to_tasks,
        domain_to_encoders=DOMAIN_TO_ENCODERS,
        test_task_ratio=0.2,
        seed=args.seed,
        fold=args.fold,
        num_folds=args.num_folds,
    )

    logger.info("Initializing UniversalTabICL model...")
    model = UniversalTabICL(
        target_dim=args.target_dim,
        lora_r=args.lora_r,
        enable_lora=True,
        device=args.device,
        # Gradient checkpointing only pays off across a backward pass; this
        # script is inference-only (forward_episodic_ensemble runs under
        # torch.no_grad()) and checkpointing under no_grad is unsupported by
        # TabICL's recompute implementation, so it must stay off here.
        recompute=False,
    )

    if args.checkpoint:
        logger.info(f"Loading LoRA weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    model.eval()

    all_regime_summaries = {}

    for regime in regimes:
        logger.info(f"========== EVALUATING REGIME: {regime.upper()} ==========")
        pairs = splits_manager.get_eval_pairs(regime=regime, holdout_domain=args.holdout_domain)
        logger.info(f"Found {len(pairs)} evaluation pairs for regime '{regime}'")

        results = {}
        metric_lists = {"accuracy": [], "balanced_accuracy": [], "roc_auc": [], "auprc": [], "macro_f1": [], "mcc": []}

        regime_t0 = time.time()
        for pair_idx, (domain, task_id, enc_id) in enumerate(pairs, start=1):
            pair_t0 = time.time()
            m = evaluate_pair(
                model,
                data_root,
                domain,
                task_id,
                enc_id,
                device=args.device,
                num_projections=args.num_projections,
            )
            pair_elapsed = time.time() - pair_t0
            elapsed = time.time() - regime_t0
            avg_per_pair = elapsed / pair_idx
            eta = avg_per_pair * (len(pairs) - pair_idx)
            if m:
                score_str = (
                    f"Acc={m.get('accuracy', float('nan')):.4f} "
                    f"BalAcc={m.get('balanced_accuracy', float('nan')):.4f} "
                    f"AUROC={m.get('roc_auc', float('nan')):.4f} "
                    f"AUPRC={m.get('auprc', float('nan')):.4f} "
                    f"F1={m.get('macro_f1', float('nan')):.4f} "
                    f"MCC={m.get('mcc', float('nan')):.4f}"
                )
            else:
                score_str = "SKIPPED (no valid data)"
            logger.info(
                f"[{regime}] pair {pair_idx}/{len(pairs)} {domain}/{task_id}/{enc_id} "
                f"{score_str} | took {pair_elapsed:.1f}s (avg {avg_per_pair:.1f}s/pair, ETA {eta/60:.1f} min)"
            )
            if not m:
                continue
            key = f"{domain}::{task_id}::{enc_id}"
            results[key] = m
            for name, lst in metric_lists.items():
                v = m.get(name)
                if v is not None and not np.isnan(v):
                    lst.append(v)

        regime_summary = {
            "n_evaluated": len(results),
            **{f"mean_{name}": float(np.mean(lst)) if lst else 0.0 for name, lst in metric_lists.items()},
            "results": results,
        }
        all_regime_summaries[regime] = regime_summary

        logger.info(
            f"--> [{regime.upper()}] Evaluated {len(results)} pairs | "
            f"Mean Acc: {regime_summary['mean_accuracy']:.4f} | "
            f"Mean AUROC: {regime_summary['mean_roc_auc']:.4f} | "
            f"Mean F1: {regime_summary['mean_macro_f1']:.4f}"
        )

    out_file = out_dir / "universal_generalization_summary.json"
    with open(out_file, "w") as f:
        json.dump(all_regime_summaries, f, indent=2)
    logger.info(f"Saved complete universal evaluation summary to {out_file}")


if __name__ == "__main__":
    main()
