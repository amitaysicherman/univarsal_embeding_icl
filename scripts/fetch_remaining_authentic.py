#!/usr/bin/env python3
"""Fetch remaining authentic protein and audio benchmark datasets with ZERO synthetic fallbacks."""

import json
import logging
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np
import wave
import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fetch_remaining")


def save_task(
    out_dir: Path,
    domain: str,
    task_id: str,
    inputs: List[Any],
    labels: np.ndarray,
    split_type: str,
    description: str,
    source: str,
    class_names: List[str],
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> bool:
    task_dir = out_dir / "tasks" / domain / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(labels)
    if n_samples < 30:
        logger.warning(f"Task '{task_id}' has too few samples ({n_samples}), skipping.")
        return False

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        logger.warning(f"Task '{task_id}' has only {len(classes)} class, skipping.")
        return False

    rng = np.random.default_rng(42)
    folds = np.zeros(n_samples, dtype=np.int8)  # 0=test, 1=val, 2=train

    for c in classes:
        c_idx = np.flatnonzero(labels == c)
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_te = max(1, int(n_c * split_ratios[2]))
        n_va = max(1, int(n_c * split_ratios[1]))

        te_idx = c_idx[:n_te]
        va_idx = c_idx[n_te : n_te + n_va]
        tr_idx = c_idx[n_te + n_va :]

        folds[te_idx] = 0
        folds[va_idx] = 1
        folds[tr_idx] = 2

    label_map = {c: i for i, c in enumerate(classes)}
    mapped_labels = np.array([label_map[lbl] for lbl in labels], dtype=np.int64)

    np.save(task_dir / "labels.npy", mapped_labels)
    np.save(task_dir / "folds.npy", folds)

    meta = {
        "task_id": task_id,
        "domain": domain,
        "num_classes": len(classes),
        "class_names": class_names[: len(classes)],
        "num_samples": n_samples,
        "split_type": split_type,
        "source": source,
        "description": description,
        "class_distribution": counts.tolist(),
        "folds_distribution": {
            "test": int((folds == 0).sum()),
            "val": int((folds == 1).sum()),
            "train": int((folds == 2).sum()),
        },
    }
    with open(task_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(task_dir / "inputs.json", "w") as f:
        json.dump(inputs, f)

    logger.info(
        f"✓ Successfully saved REAL task [{domain}] '{task_id}': "
        f"N={n_samples} | {len(classes)} classes | Source: {source}"
    )
    return True


def fetch_protein_tasks(out_dir: Path, max_samples: int = 1500) -> int:
    logger.info("--> Fetching REAL Protein sequence benchmarks from SaProtHub...")
    prot_specs = [
        (
            "prot_deeploc_binary",
            "SaProtHub/Dataset-Binary_Localization-DeepLoc",
            ["membrane", "soluble"],
            "DeepLoc protein membrane vs soluble localization",
        ),
        (
            "prot_deeploc_multi",
            "SaProtHub/Dataset-Subcellular_Localization-DeepLoc",
            [
                "Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum", "Golgi.apparatus",
                "Lysosome", "Mitochondrion", "Nucleus", "Peroxisome", "Plastid", "Extracellular"
            ],
            "DeepLoc 10-class subcellular localization",
        ),
        (
            "prot_metal_ion",
            "SaProtHub/Dataset-Metal_Ion_Binding",
            ["non_metal", "metal_binding"],
            "Protein metal ion binding prediction",
        ),
    ]
    saved = 0
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for task_id, hf_id, cnames, desc in prot_specs:
        try:
            ds = datasets.load_dataset(hf_id, split=f"train[:{max_samples}]")
            clean_seqs, labels = [], []
            for row in ds:
                raw_seq = row["protein"]
                seq = "".join([c for c in raw_seq if c.isupper() and c in valid_aa])
                if len(seq) >= 10:
                    clean_seqs.append(seq)
                    labels.append(row["label"])
            labels_np = np.array(labels, dtype=np.int64)
            if save_task(out_dir, "proteins", task_id, clean_seqs, labels_np, "stratified_benchmark_split", desc, f"HuggingFace ({hf_id})", cnames):
                saved += 1
        except Exception as e:
            logger.error(f"Failed to fetch protein task '{task_id}': {e}")
    return saved


def fetch_audio_tasks(out_dir: Path, raw_audio_dir: Path) -> int:
    logger.info("--> Fetching REAL Audio benchmarks from ESC-50...")
    saved = 0
    try:
        ds = datasets.load_dataset("ashraq/esc50", split="train")
        
        # 1. ESC-10 benchmark (400 samples, 10 classes)
        esc10_dir = raw_audio_dir / "audio_esc10"
        esc10_dir.mkdir(parents=True, exist_ok=True)
        esc10_paths, esc10_labels, esc10_cnames_map = [], [], {}

        # 2. ESC-50 full benchmark (2000 samples, 50 classes)
        esc50_dir = raw_audio_dir / "audio_esc50"
        esc50_dir.mkdir(parents=True, exist_ok=True)
        esc50_paths, esc50_labels, esc50_cnames_map = [], [], {}

        for i, row in enumerate(ds):
            arr = row["audio"]["array"]
            sr = row["audio"]["sampling_rate"]
            sig = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
            cat = row["category"]
            target = row["target"]

            # Save full ESC-50
            fpath50 = esc50_dir / f"esc50_{i:05d}.wav"
            if not fpath50.exists():
                with wave.open(str(fpath50), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(sig.tobytes())
            esc50_paths.append(str(fpath50))
            esc50_labels.append(target)
            esc50_cnames_map[target] = cat

            # If part of ESC-10
            if row["esc10"]:
                fpath10 = esc10_dir / f"esc10_{len(esc10_paths):05d}.wav"
                if not fpath10.exists():
                    with wave.open(str(fpath10), "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(sig.tobytes())
                esc10_paths.append(str(fpath10))
                esc10_labels.append(target)
                esc10_cnames_map[target] = cat

        # Save ESC-10
        esc10_labels_np = np.array(esc10_labels, dtype=np.int64)
        sorted_targets_10 = sorted(esc10_cnames_map.keys())
        cnames_10 = [esc10_cnames_map[t] for t in sorted_targets_10]
        if save_task(
            out_dir, "audio", "audio_esc10", esc10_paths, esc10_labels_np,
            "official_esc_split", "ESC-10 environmental sound classification (10 classes)",
            "HuggingFace (ashraq/esc50)", cnames_10
        ):
            saved += 1

        # Save ESC-50
        esc50_labels_np = np.array(esc50_labels, dtype=np.int64)
        sorted_targets_50 = sorted(esc50_cnames_map.keys())
        cnames_50 = [esc50_cnames_map[t] for t in sorted_targets_50]
        if save_task(
            out_dir, "audio", "audio_esc50", esc50_paths, esc50_labels_np,
            "official_esc_split", "ESC-50 environmental sound classification (50 classes)",
            "HuggingFace (ashraq/esc50)", cnames_50
        ):
            saved += 1

    except Exception as e:
        logger.error(f"Failed to fetch audio tasks from ESC-50: {e}")
    return saved


def main():
    data_root = Path("data")
    raw_audio_dir = data_root / "raw" / "audio"

    logger.info("FETCHING REMAINING AUTHENTIC BENCHMARK TASKS (PROTEINS & AUDIO)")
    n_prot = fetch_protein_tasks(data_root)
    n_audio = fetch_audio_tasks(data_root, raw_audio_dir)
    logger.info(f"DONE: Proteins={n_prot}, Audio={n_audio}")


if __name__ == "__main__":
    main()
