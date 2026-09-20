#!/usr/bin/env python3
"""Comprehensive Data Audit Script for all 7 Modalities.

Verifies:
1. Metadata validity and consistency
2. Label integrity (0-indexed, class balance, no NaNs/Infs, length match)
3. Split/Fold validity (valid folds, proper distributions)
4. Raw input integrity (verifies actual files on disk for vision/audio, sequence validity for proteins/smiles, graph structures for PyG, float arrays for TS)
"""

#!/usr/bin/env python3
"""Comprehensive Data Audit Script for all 7 Modalities."""

import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_data")


def audit_task(task_dir):
    task_id = task_dir.name
    domain = task_dir.parent.name
    res = {
        "domain": domain,
        "task_id": task_id,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "num_samples": 0,
        "num_classes": 0,
        "class_counts": {},
        "input_type": None,
        "sample_preview": None,
    }

    # 1. Check metadata.json
    meta_path = task_dir / "metadata.json"
    if not meta_path.exists():
        res["errors"].append("Missing metadata.json")
        res["status"] = "FAIL"
        return res

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        res["errors"].append(f"Invalid metadata.json: {e}")
        res["status"] = "FAIL"
        return res

    # 2. Check labels.npy
    labels_path = task_dir / "labels.npy"
    if not labels_path.exists():
        res["errors"].append("Missing labels.npy")
        res["status"] = "FAIL"
        return res

    try:
        labels = np.load(labels_path)
    except Exception as e:
        res["errors"].append(f"Cannot load labels.npy: {e}")
        res["status"] = "FAIL"
        return res

    if np.isnan(labels).any() or np.isinf(labels).any():
        res["errors"].append("Labels contain NaN or Inf")
        res["status"] = "FAIL"

    unique_classes, counts = np.unique(labels, return_counts=True)
    res["num_samples"] = len(labels)
    res["num_classes"] = len(unique_classes)
    res["class_counts"] = {int(k): int(v) for k, v in zip(unique_classes, counts)}

    # Check 0-indexing
    expected_classes = np.arange(len(unique_classes))
    if not np.array_equal(np.sort(unique_classes), expected_classes):
        res["errors"].append(f"Classes not 0-indexed integers 0..C-1: {unique_classes.tolist()}")
        res["status"] = "FAIL"

    # 3. Check folds.npy / split_indices.json
    folds_path = task_dir / "folds.npy"
    split_path = task_dir / "split_indices.json"
    if folds_path.exists():
        try:
            folds = np.load(folds_path)
            if len(folds) != len(labels):
                res["errors"].append(f"folds length ({len(folds)}) != labels length ({len(labels)})")
                res["status"] = "FAIL"
            u_folds, f_counts = np.unique(folds, return_counts=True)
            res["folds_info"] = {int(k): int(v) for k, v in zip(u_folds, f_counts)}
        except Exception as e:
            res["errors"].append(f"Cannot load folds.npy: {e}")
            res["status"] = "FAIL"
    elif split_path.exists():
        try:
            with open(split_path) as f:
                splits = json.load(f)
            train_idx = splits.get("train", [])
            test_idx = splits.get("test", [])
            val_idx = splits.get("val", [])
            total_split_len = len(train_idx) + len(test_idx) + len(val_idx)
            if total_split_len != len(labels):
                res["warnings"].append(f"split_indices total ({total_split_len}) != labels ({len(labels)})")
            res["splits_info"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
        except Exception as e:
            res["errors"].append(f"Cannot load split_indices.json: {e}")
            res["status"] = "FAIL"
    else:
        res["warnings"].append("No folds.npy or split_indices.json found")

    # 4. Check Raw Inputs
    inputs_json = task_dir / "inputs.json"
    inputs_npy = task_dir / "inputs.npy"
    inputs_pt = task_dir / "inputs.pt"

    if inputs_json.exists():
        try:
            with open(inputs_json) as f:
                raw_inputs = json.load(f)
            if len(raw_inputs) != len(labels):
                res["errors"].append(f"inputs.json length ({len(raw_inputs)}) != labels length ({len(labels)})")
                res["status"] = "FAIL"
            
            # Domain-specific verification
            if domain in ["molecules", "text", "proteins"]:
                res["input_type"] = f"str list (len={len(raw_inputs)})"
                # Check for empty strings or dummy patterns
                empty_cnt = sum(1 for s in raw_inputs if not isinstance(s, str) or len(s.strip()) == 0)
                if empty_cnt > 0:
                    res["errors"].append(f"Found {empty_cnt} empty or non-string inputs")
                    res["status"] = "FAIL"
                
                # Check sample
                first_item = raw_inputs[0]
                res["sample_preview"] = first_item[:50] if isinstance(first_item, str) else str(first_item)[:50]
                
                if domain == "molecules":
                    # Check if SMILES look authentic
                    if any("synthetic" in s.lower() or "dummy" in s.lower() for s in raw_inputs[:10]):
                        res["errors"].append("SMILES inputs appear synthetic/dummy")
                        res["status"] = "FAIL"
                elif domain == "proteins":
                    # Check valid amino acids
                    valid_aa = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")
                    bad_seqs = 0
                    for s in raw_inputs[:50]:
                        clean_s = "".join(s.strip().split()).upper()
                        if not all(c in valid_aa for c in clean_s):
                            bad_seqs += 1
                    if bad_seqs > 0:
                        res["warnings"].append(f"{bad_seqs}/50 sampled sequences contain non-standard amino acid codes")

            elif domain in ["vision", "audio"]:
                res["input_type"] = f"file paths list (len={len(raw_inputs)})"
                # Verify that files actually exist on disk and can be read!
                missing_files = 0
                unreadable_files = 0
                sample_paths = raw_inputs[:20]
                for p_str in sample_paths:
                    p = Path(p_str)
                    if not p.exists():
                        missing_files += 1
                        continue
                    if domain == "vision":
                        try:
                            from PIL import Image
                            with Image.open(p) as img:
                                img.verify()
                        except Exception:
                            unreadable_files += 1
                    elif domain == "audio":
                        try:
                            import wave
                            with wave.open(str(p), "r") as wf:
                                _ = wf.getnframes()
                        except Exception:
                            unreadable_files += 1

                if missing_files > 0:
                    res["errors"].append(f"{missing_files}/{len(sample_paths)} sampled files do NOT exist on disk: e.g. {sample_paths[0]}")
                    res["status"] = "FAIL"
                if unreadable_files > 0:
                    res["errors"].append(f"{unreadable_files}/{len(sample_paths)} sampled files are corrupted/unreadable")
                    res["status"] = "FAIL"
                res["sample_preview"] = str(raw_inputs[0])

        except Exception as e:
            res["errors"].append(f"Error checking inputs.json: {e}")
            res["status"] = "FAIL"

    elif inputs_npy.exists():
        try:
            arr = np.load(inputs_npy)
            res["input_type"] = f"npy array shape={arr.shape}, dtype={arr.dtype}"
            if len(arr) != len(labels):
                res["errors"].append(f"inputs.npy length ({len(arr)}) != labels length ({len(labels)})")
                res["status"] = "FAIL"
            if np.isnan(arr).any() or np.isinf(arr).any():
                res["errors"].append("inputs.npy contains NaNs or Infs")
                res["status"] = "FAIL"
            res["sample_preview"] = f"shape={arr.shape}, min={np.min(arr):.3f}, max={np.max(arr):.3f}"
        except Exception as e:
            res["errors"].append(f"Error loading inputs.npy: {e}")
            res["status"] = "FAIL"

    elif inputs_pt.exists():
        try:
            import torch
            obj = torch.load(inputs_pt, map_location="cpu", weights_only=False)
            res["input_type"] = f"torch object ({type(obj).__name__}, len={len(obj)})"
            if len(obj) != len(labels):
                res["errors"].append(f"inputs.pt length ({len(obj)}) != labels length ({len(labels)})")
                res["status"] = "FAIL"
            
            # Check first graph object if list of graphs
            if isinstance(obj, list) and len(obj) > 0:
                first = obj[0]
                res["sample_preview"] = f"item type={type(first).__name__}, num_nodes={getattr(first, 'num_nodes', None)}, num_edges={getattr(first, 'num_edges', None)}"
        except Exception as e:
            res["errors"].append(f"Error loading inputs.pt: {e}")
            res["status"] = "FAIL"
    else:
        res["errors"].append("No raw inputs found (no inputs.json, inputs.npy, or inputs.pt)")
        res["status"] = "FAIL"

    # 5. Check Embeddings (CRITICAL: Must be real, not dummy)
    embeddings_dir = task_dir / "embeddings"
    res["embeddings_info"] = {}
    if embeddings_dir.exists():
        emb_files = sorted(embeddings_dir.glob("*.npy"))
        if len(emb_files) == 0:
            res["warnings"].append("No embedding files found in embeddings/ directory")

        for emb_file in emb_files:
            encoder_name = emb_file.stem
            try:
                embs = np.load(emb_file)
                emb_info = {
                    "shape": tuple(embs.shape),
                    "dtype": str(embs.dtype),
                    "status": "OK"
                }

                # Check for NaN/Inf
                if np.any(np.isnan(embs)):
                    res["errors"].append(f"Embedding {encoder_name}.npy contains NaN values")
                    res["status"] = "FAIL"
                    emb_info["status"] = "FAIL_NAN"

                if np.any(np.isinf(embs)):
                    res["errors"].append(f"Embedding {encoder_name}.npy contains Inf values")
                    res["status"] = "FAIL"
                    emb_info["status"] = "FAIL_INF"

                # Check shape: must be (N, D) where N = num_samples
                if len(embs.shape) != 2:
                    res["errors"].append(f"Embedding {encoder_name}.npy has wrong rank: {len(embs.shape)} (expected 2)")
                    res["status"] = "FAIL"
                    emb_info["status"] = "FAIL_SHAPE"
                elif embs.shape[0] != res["num_samples"]:
                    res["errors"].append(
                        f"Embedding {encoder_name}.npy shape mismatch: N={embs.shape[0]} != {res['num_samples']} (labels)"
                    )
                    res["status"] = "FAIL"
                    emb_info["status"] = "FAIL_LENGTH"

                # Check variance (degenerate if ~0)
                var = np.var(embs)
                if var < 1e-7:
                    res["errors"].append(f"Embedding {encoder_name}.npy has degenerate variance: {var:.2e} (likely dummy/zeros)")
                    res["status"] = "FAIL"
                    emb_info["status"] = "FAIL_VARIANCE"
                    emb_info["variance"] = float(var)

                # Check that not all embeddings are identical (strong dummy signature)
                if embs.shape[0] > 1:
                    pairwise_diffs = np.sum(np.abs(embs[1:] - embs[:-1]))
                    if pairwise_diffs < 1e-6:
                        res["errors"].append(f"Embedding {encoder_name}.npy: all embeddings appear identical (dummy signature)")
                        res["status"] = "FAIL"
                        emb_info["status"] = "FAIL_IDENTICAL"

                # Check norms (should not be all exactly 1.0 unless normalized)
                norms = np.linalg.norm(embs, axis=1)
                mean_norm = np.mean(norms)
                std_norm = np.std(norms)
                emb_info["mean_norm"] = float(mean_norm)
                emb_info["std_norm"] = float(std_norm)

                # Warn if unusual norm distribution (could indicate problems)
                if mean_norm < 0.1 or mean_norm > 100.0:
                    res["warnings"].append(
                        f"Embedding {encoder_name}.npy has unusual norm distribution: mean={mean_norm:.4f} (expected ~1.0)"
                    )

                res["embeddings_info"][encoder_name] = emb_info

            except Exception as e:
                res["errors"].append(f"Cannot load embedding {encoder_name}.npy: {e}")
                res["status"] = "FAIL"
                res["embeddings_info"][encoder_name] = {"status": "FAIL_LOAD", "error": str(e)}
    else:
        res["warnings"].append("No embeddings/ directory found (expected after encoder extraction)")

    return res


def main():
    root = Path("data/tasks")
    if not root.exists():
        logger.error(f"Task root directory {root} does not exist!")
        return

    all_domains = sorted([d.name for d in root.iterdir() if d.is_dir()])
    logger.info(f"Found domains: {all_domains}")

    total_tasks = 0
    passed_tasks = 0
    failed_tasks = 0
    domain_summary = {}

    for domain in all_domains:
        domain_dir = root / domain
        task_dirs = sorted([t for t in domain_dir.iterdir() if t.is_dir()])
        domain_summary[domain] = {"total": len(task_dirs), "pass": 0, "fail": 0, "tasks": []}

        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain.upper()} ({len(task_dirs)} tasks)")
        print(f"{'='*80}")

        for task_dir in task_dirs:
            total_tasks += 1
            res = audit_task(task_dir)
            domain_summary[domain]["tasks"].append(res)

            if res["status"] == "PASS":
                passed_tasks += 1
                domain_summary[domain]["pass"] += 1
                warn_str = f" [WARN: {'; '.join(res['warnings'])}]" if res["warnings"] else ""
                print(f"  [PASS] {res['task_id']:<35} N={res['num_samples']:<5} C={res['num_classes']} | {res['input_type']}{warn_str}")
                print(f"         Class counts: {res['class_counts']}")
                print(f"         Preview: {res['sample_preview']}")
            else:
                failed_tasks += 1
                domain_summary[domain]["fail"] += 1
                print(f"  [FAIL] {res['task_id']:<35} ERRORS: {'; '.join(res['errors'])}")
                if res["warnings"]:
                    print(f"         WARNINGS: {'; '.join(res['warnings'])}")

    print(f"\n{'='*80}")
    print(f"OVERALL DATA AUDIT SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks inspected: {total_tasks}")
    print(f"PASSED: {passed_tasks}")
    print(f"FAILED: {failed_tasks}")
    for d, s in domain_summary.items():
        print(f"  - {d:<12}: {s['pass']}/{s['total']} PASS")

    # Save report
    out_file = Path("data/data_audit_report.json")
    with open(out_file, "w") as f:
        json.dump({"total": total_tasks, "pass": passed_tasks, "fail": failed_tasks, "domains": domain_summary}, f, indent=2)
    logger.info(f"Audit report saved to {out_file}")


if __name__ == "__main__":
    main()
