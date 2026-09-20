#!/usr/bin/env python3
"""Meta-training script for Universal TabICL with Random Projection and LoRA Adaptation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from univarsal_embeding.encoders.registry import DOMAIN_TO_ENCODERS
from univarsal_embeding.eval.splits import UniversalBenchmarkSplits
from univarsal_embeding.models.tabicl_universal import UniversalTabICL
from evaluate_universal import evaluate_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_tabicl_lora")


class EpisodicMetaDataset:
    """Pre-indexes tasks, embeddings, labels, and folds for ultra-fast episodic sampling."""

    def __init__(self, data_root: Path, train_pairs: List[Tuple[str, str, str]], device: str = "cpu") -> None:
        self.device = device
        self.episodes: List[Dict[str, Any]] = []

        logger.info(f"Loading {len(train_pairs)} train (domain, task, encoder) triples into memory...")
        for domain, task_id, enc_id in train_pairs:
            task_dir = data_root / "tasks" / domain / task_id
            emb_path = task_dir / "embeddings" / f"{enc_id}.npy"
            labels_path = task_dir / "labels.npy"
            folds_path = task_dir / "folds.npy"

            if not (emb_path.exists() and labels_path.exists() and folds_path.exists()):
                continue

            X = np.load(emb_path)
            y = np.load(labels_path)
            folds = np.load(folds_path)

            train_idx = np.flatnonzero(folds == 2)
            val_idx = np.flatnonzero(folds == 1)

            if len(val_idx) == 0:
                n_val = max(1, int(len(train_idx) * 0.2))
                val_idx = train_idx[:n_val]
                train_idx = train_idx[n_val:]

            if len(train_idx) < 10 or len(val_idx) < 5 or len(np.unique(y[train_idx])) < 2:
                continue

            self.episodes.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "encoder": enc_id,
                    "X": torch.tensor(X, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.long),
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "n_classes": len(np.unique(y[train_idx])),
                }
            )

        logger.info(f"Successfully indexed {len(self.episodes)} valid meta-training episodes.")

    def sample_episode(
        self,
        max_context: int | None = None,
        max_query: int = 64,
        rng: np.random.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if rng is None:
            rng = np.random.default_rng()

        item = self.episodes[rng.integers(0, len(self.episodes))]
        X, y = item["X"], item["y"]
        train_idx, val_idx = item["train_idx"], item["val_idx"]

        # Context = ALL available training samples for this task/encoder, never
        # subsampled - matches official TabICL usage (the full training set is
        # always used as in-context examples; only the ensemble/query dimension
        # is ever batched for memory, never the context row count).
        ctx_chosen = train_idx if max_context is None else train_idx[: max_context]

        # Sample query from val_idx (this is a normal minibatch over the loss
        # targets, not a context clip, so remains bounded by max_query)
        n_q = min(max_query, len(val_idx))
        q_chosen = rng.choice(val_idx, size=n_q, replace=False)

        # Remap labels locally to 0..C-1
        ctx_y_raw = y[ctx_chosen].numpy()
        classes, inv = np.unique(ctx_y_raw, return_inverse=True)
        label_map = {c: i for i, c in enumerate(classes)}

        y_ctx = torch.tensor(inv, dtype=torch.long, device=self.device)
        y_q_raw = y[q_chosen].numpy()
        y_q_mapped = np.array([label_map.get(lbl, 0) for lbl in y_q_raw])
        y_query = torch.tensor(y_q_mapped, dtype=torch.long, device=self.device)

        X_ctx = X[ctx_chosen].to(self.device)
        X_query = X[q_chosen].to(self.device)

        return X_ctx, y_ctx, X_query, y_query


def run_eval_pairs(model, data_root, pairs, device, num_projections):
    """Runs evaluate_pair over a list of (domain, task, encoder) pairs and
    returns (mean_balanced_accuracy, mean_accuracy, mean_roc_auc, n_evaluated).
    Model is switched to eval() for the duration and restored to train() after.
    """
    was_training = model.training
    model.eval()
    bal_accs, accs, aurocs = [], [], []
    with torch.no_grad():
        for domain, task_id, enc_id in pairs:
            m = evaluate_pair(model, data_root, domain, task_id, enc_id, device=device, num_projections=num_projections)
            if not m:
                continue
            accs.append(m["accuracy"])
            bal_accs.append(m["balanced_accuracy"])
            if not np.isnan(m.get("roc_auc", float("nan"))):
                aurocs.append(m["roc_auc"])
    if was_training:
        model.train()
    n = len(bal_accs)
    return (
        float(np.mean(bal_accs)) if bal_accs else 0.0,
        float(np.mean(accs)) if accs else 0.0,
        float(np.mean(aurocs)) if aurocs else float("nan"),
        n,
    )


def train_meta_lora(args):
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

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
        test_task_ratio=args.test_task_ratio,
        seed=args.seed,
        fold=args.fold,
        num_folds=args.num_folds,
    )

    # If Leave-Domain(s)-Out: train only on domains not in holdout_domain (a list,
    # possibly of more than one domain for grouped leave-N-domains-out runs).
    if args.holdout_domain:
        logger.info(f"Leave-Domain(s)-Out mode: Holding out domains {args.holdout_domain}")
        train_pairs = []
        for d in splits_manager.domains:
            if d not in args.holdout_domain:
                part = splits_manager.partitions[d]
                for t in part.train_tasks:
                    for e in part.train_encoders:
                        train_pairs.append((d, t, e))
    else:
        train_pairs = splits_manager.get_eval_pairs(regime="seen")

    meta_dataset = EpisodicMetaDataset(data_root, train_pairs, device=device)
    if len(meta_dataset.episodes) == 0:
        logger.error("No valid episodes found to train on. Exiting.")
        return

    # Early-stopping signal: for LODO runs, the held-out domain IS the real
    # target, so early-stop on unseen_domain regardless of --early-stop-regime.
    # For regular runs, use the configured regime (default unseen_tasks - a
    # genuine held-out-task generalization signal, not train loss).
    if args.holdout_domain:
        early_stop_pairs = splits_manager.get_eval_pairs(regime="unseen_domain", holdout_domain=args.holdout_domain)
        early_stop_regime_name = f"unseen_domain({'+'.join(args.holdout_domain)})"
    else:
        early_stop_pairs = splits_manager.get_eval_pairs(regime=args.early_stop_regime)
        early_stop_regime_name = args.early_stop_regime
    logger.info(
        f"Early-stopping signal: regime='{early_stop_regime_name}' ({len(early_stop_pairs)} pairs), "
        f"eval every {args.eval_interval} steps, patience={args.patience}, "
        f"num_projections={args.eval_num_projections} (fast/approximate during training)"
    )

    logger.info("Initializing UniversalTabICL model with LoRA...")
    model = UniversalTabICL(
        target_dim=args.target_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        enable_lora=True,
        device=device,
    )

    # Collect trainable parameters (LoRA + learnable RP if enabled)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_steps, eta_min=args.lr * 0.05)

    rng = np.random.default_rng(args.seed)
    model.train()

    t0 = time.time()
    logger.info(f"Starting meta-training for up to {args.meta_steps} steps (early stopping enabled)...")

    best_score = -1.0
    best_step = 0
    no_improve_evals = 0
    stopped_early = False
    best_ckpt_path = out_dir / "tabicl_lora_best.pt"

    running_loss = 0.0
    final_step = args.meta_steps
    for step in range(1, args.meta_steps + 1):
        X_ctx, y_ctx, X_query, y_query = meta_dataset.sample_episode(
            max_context=args.max_context, max_query=args.max_query, rng=rng
        )

        optimizer.zero_grad()
        try:
            logits = model.forward_episodic(
                X_ctx, y_ctx, X_query,
                dynamic_rp=True,
                projection_seed=None,
                learnable_rp=args.learnable_rp
            )
            loss = F.cross_entropy(logits, y_query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        except Exception as e:
            logger.warning(f"Meta-step {step} forward/backward failed: {e}")
            optimizer.zero_grad()
            continue

        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t0
            logger.info(f"Step [{step}/{args.meta_steps}] Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {elapsed:.1f}s")
            running_loss = 0.0

        if step % args.save_interval == 0 or step == args.meta_steps:
            save_path = out_dir / f"tabicl_lora_step{step}.pt"
            lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k or "proj_" in k}
            torch.save(
                {
                    "step": step,
                    "state_dict": lora_state,
                    "target_dim": args.target_dim,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                },
                save_path,
            )
            logger.info(f"Saved LoRA checkpoint to {save_path}")

        if step % args.eval_interval == 0:
            eval_t0 = time.time()
            bal_acc, acc, auroc, n_eval = run_eval_pairs(
                model, data_root, early_stop_pairs, device, args.eval_num_projections
            )
            logger.info(
                f"[early-stop check @ step {step}] regime={early_stop_regime_name} n={n_eval} "
                f"BalAcc={bal_acc:.4f} Acc={acc:.4f} AUROC={auroc:.4f} (took {time.time()-eval_t0:.1f}s)"
            )
            if bal_acc > best_score:
                best_score = bal_acc
                best_step = step
                no_improve_evals = 0
                lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k or "proj_" in k}
                torch.save(
                    {
                        "step": step,
                        "state_dict": lora_state,
                        "target_dim": args.target_dim,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "early_stop_score": best_score,
                        "early_stop_regime": early_stop_regime_name,
                    },
                    best_ckpt_path,
                )
                logger.info(f"New best {early_stop_regime_name} BalAcc={best_score:.4f} @ step {step} -> saved {best_ckpt_path}")
            else:
                no_improve_evals += 1
                logger.info(
                    f"No improvement ({no_improve_evals}/{args.patience}) - best BalAcc={best_score:.4f} @ step {best_step}"
                )
                if no_improve_evals >= args.patience:
                    logger.info(
                        f"Early stopping at step {step}: no improvement on '{early_stop_regime_name}' "
                        f"for {args.patience} consecutive evals ({args.patience * args.eval_interval} steps)."
                    )
                    final_step = step
                    stopped_early = True
                    break

    logger.info(
        f"Meta-training finished ({'early-stopped' if stopped_early else 'reached meta-steps limit'}) "
        f"at step {final_step}. Best {early_stop_regime_name} BalAcc={best_score:.4f} @ step {best_step}."
    )

    # Chain the full final evaluation (all requested regimes, full ensemble
    # size) right into this same run - no separate script/job needed. Uses
    # the BEST checkpoint (by held-out balanced accuracy), not just the last one.
    if best_ckpt_path.exists():
        logger.info(f"Loading best checkpoint (step {best_step}) for final evaluation...")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"], strict=False)
    model.eval()

    final_regimes = [r.strip() for r in args.final_eval_regimes.split(",")] if args.final_eval_regimes else []
    if final_regimes:
        logger.info(f"Running FULL final evaluation on regimes: {final_regimes} (num_projections={args.num_projections})")
        all_summaries = {}
        for regime in final_regimes:
            pairs = splits_manager.get_eval_pairs(regime=regime, holdout_domain=args.holdout_domain)
            logger.info(f"========== FINAL EVAL REGIME: {regime.upper()} ({len(pairs)} pairs) ==========")
            metric_lists = {"accuracy": [], "balanced_accuracy": [], "roc_auc": [], "auprc": [], "macro_f1": [], "mcc": []}
            results = {}
            regime_t0 = time.time()
            with torch.no_grad():
                for pair_idx, (domain, task_id, enc_id) in enumerate(pairs, start=1):
                    m = evaluate_pair(model, data_root, domain, task_id, enc_id, device=device, num_projections=args.num_projections)
                    if not m:
                        continue
                    key = f"{domain}::{task_id}::{enc_id}"
                    results[key] = m
                    for name, lst in metric_lists.items():
                        v = m.get(name)
                        if v is not None and not np.isnan(v):
                            lst.append(v)
                    if pair_idx % 20 == 0 or pair_idx == len(pairs):
                        logger.info(f"[final-eval:{regime}] {pair_idx}/{len(pairs)} pairs done ({time.time()-regime_t0:.0f}s elapsed)")
            summary = {
                "n_evaluated": len(results),
                **{f"mean_{name}": float(np.mean(lst)) if lst else 0.0 for name, lst in metric_lists.items()},
                "results": results,
            }
            all_summaries[regime] = summary
            logger.info(
                f"--> [{regime.upper()}] Evaluated {len(results)} pairs | Mean Acc: {summary['mean_accuracy']:.4f} | "
                f"Mean BalAcc: {summary['mean_balanced_accuracy']:.4f} | Mean AUROC: {summary['mean_roc_auc']:.4f} | "
                f"Mean AUPRC: {summary['mean_auprc']:.4f} | Mean MCC: {summary['mean_mcc']:.4f}"
            )
        summary_path = out_dir / "final_evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "best_step": best_step,
                    "final_step": final_step,
                    "stopped_early": stopped_early,
                    "early_stop_regime": early_stop_regime_name,
                    "early_stop_best_score": best_score,
                    "regimes": all_summaries,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved full final evaluation summary to {summary_path}")

    logger.info("Meta-training + final evaluation finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Universal TabICL LoRA Meta-Training")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/universal_tabicl")
    parser.add_argument("--meta-steps", type=int, default=1000)
    parser.add_argument("--max-context", type=int, default=None, help="If set, caps context size; default None uses ALL training samples (matches official TabICL usage)")
    parser.add_argument("--max-query", type=int, default=64)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learnable-rp", action="store_true")
    parser.add_argument(
        "--holdout-domain", type=str, default=None,
        help="For Leave-Domain(s)-Out meta-training. Comma-separated for grouped holdout, e.g. 'audio,graphs'",
    )
    parser.add_argument("--test-task-ratio", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--fold", type=int, default=None, help="Fold index (0-4) for 5-fold cross-generalization")
    parser.add_argument("--num-folds", type=int, default=5, help="Total number of folds for cross-generalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-interval", type=int, default=500, help="Meta-steps between held-out early-stopping evaluations")
    parser.add_argument("--patience", type=int, default=3, help="Consecutive non-improving evals before stopping early")
    parser.add_argument(
        "--early-stop-regime", type=str, default="seen",
        help="Regime used for the early-stopping signal (ignored for LODO runs, which always use unseen_domain)",
    )
    parser.add_argument(
        "--eval-num-projections", type=int, default=4,
        help="Ensemble size for the FAST periodic early-stopping eval (fewer projections = quicker checks during training)",
    )
    parser.add_argument(
        "--num-projections", type=int, default=16,
        help="Ensemble size for the FULL final evaluation run at the end of training",
    )
    parser.add_argument(
        "--final-eval-regimes", type=str,
        default="seen,unseen_tasks,unseen_encoders,unseen_both",
        help="Comma-separated regimes for the full final evaluation chained after training "
        "(automatically just 'unseen_domain' for LODO runs). Empty string skips final evaluation.",
    )
    args = parser.parse_args()

    if args.holdout_domain:
        args.holdout_domain = [d.strip() for d in args.holdout_domain.split(",") if d.strip()]
        if args.final_eval_regimes:
            args.final_eval_regimes = "unseen_domain"

    train_meta_lora(args)


if __name__ == "__main__":
    main()
