#!/usr/bin/env python3
"""Fetch additional real classification tasks for proteins, vision, and audio.

Criteria (per user): 2-10 classes, <=10k samples per task, has a natural
train/test-able split (we use stratified 5-fold like the rest of the
benchmark), real/authentic data only, no synthetic fallback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_extra")

OUTPUT_DIR = Path("data/tasks")
RAW_AUD_DIR = Path("data/raw/audio")


def save_task(domain, task_id, inputs, labels, class_names, source, description, input_format="json"):
    task_dir = OUTPUT_DIR / domain / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int64)
    unique_classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)

    if not np.array_equal(unique_classes, np.arange(num_classes)):
        remap = {c: i for i, c in enumerate(unique_classes)}
        labels = np.array([remap[c] for c in labels], dtype=np.int64)
        unique_classes, counts = np.unique(labels, return_counts=True)

    if num_classes < 2 or num_classes > 10:
        logger.error(f"REJECTED '{task_id}': {num_classes} classes (must be 2-10)")
        return False
    if len(labels) > 10000:
        logger.error(f"REJECTED '{task_id}': {len(labels)} samples (must be <=10000)")
        return False

    num_samples = len(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(num_samples, dtype=np.int64)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(num_samples), labels)):
        folds[test_idx] = fold_idx

    if input_format == "json":
        with open(task_dir / "inputs.json", "w") as f:
            json.dump(inputs, f)
    elif input_format == "npy":
        np.save(task_dir / "inputs.npy", np.asarray(inputs, dtype=np.float32))

    np.save(task_dir / "labels.npy", labels)
    np.save(task_dir / "folds.npy", folds)

    train_idx = np.where(folds != 0)[0].tolist()
    test_idx = np.where(folds == 0)[0].tolist()
    with open(task_dir / "split_indices.json", "w") as f:
        json.dump({"train": train_idx, "val": [], "test": test_idx}, f, indent=2)

    meta = {
        "task_id": task_id,
        "domain": domain,
        "task_type": "binary" if num_classes == 2 else "multiclass",
        "num_samples": num_samples,
        "num_classes": num_classes,
        "class_names": class_names[:num_classes] if class_names else [f"class_{i}" for i in range(num_classes)],
        "source": source,
        "description": description,
    }
    with open(task_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SAVED [{domain.upper()}] {task_id}: N={num_samples}, C={num_classes}, "
                f"classes={dict(zip(unique_classes.tolist(), counts.tolist()))}")
    return True


# ============================================================
# PROTEINS: 3 new TDC epitope/paratope real antibody tasks
# ============================================================
def fetch_extra_proteins():
    logger.info("=== Fetching extra protein tasks (TDC) ===")
    from tdc.single_pred import Epitope, Paratope

    specs = [
        ("prot_iedb_jespersen", lambda: Epitope(name="iedb_jespersen").get_data(), "Antigen", "Y",
         ["non_epitope", "epitope"], "TDC (Epitope/iedb_jespersen)",
         "IEDB experimental protein epitope recognition"),
        ("prot_pdb_jespersen", lambda: Epitope(name="pdb_jespersen").get_data(), "Antigen", "Y",
         ["non_epitope", "epitope"], "TDC (Epitope/pdb_jespersen)",
         "PDB structural protein epitope binding"),
        ("prot_sabdab_liberis", lambda: Paratope(name="sabdab_liberis").get_data(), "Antibody", "Y",
         ["non_paratope", "paratope"], "TDC (Paratope/sabdab_liberis)",
         "SAbDab antibody paratope CDR recognition"),
    ]
    for task_id, fn, seq_col, label_col, cnames, src, desc in specs:
        task_dir = OUTPUT_DIR / "proteins" / task_id
        if (task_dir / "metadata.json").exists():
            logger.info(f"'{task_id}' already exists, skipping.")
            continue
        try:
            df = fn()
            seqs = df[seq_col].astype(str).str.strip().str.upper().tolist()
            labels = df[label_col].to_numpy(dtype=np.int64)
            n = len(labels)
            if n > 8000:
                idx = np.random.RandomState(42).choice(n, 8000, replace=False)
                seqs = [seqs[i] for i in idx]
                labels = labels[idx]
            save_task("proteins", task_id, seqs, labels, cnames, src, desc, "json")
        except Exception as e:
            logger.error(f"FAILED protein task '{task_id}': {e}")


# ============================================================
# VISION: 5 new MedMNIST real medical-image tasks
# ============================================================
def fetch_extra_vision():
    logger.info("=== Fetching extra vision tasks (MedMNIST) ===")
    import medmnist
    from medmnist import INFO
    from PIL import Image
    import os

    RAW_IMG_DIR = Path("data/raw/images")
    med_tasks = [
        ("vision_dermamnist", "dermamnist", 2000, "Dermatoscopy skin lesion classification"),
        ("vision_bloodmnist", "bloodmnist", 2000, "Peripheral blood cell classification"),
        ("vision_pneumoniamnist", "pneumoniamnist", 2000, "Pediatric chest X-ray pneumonia classification"),
        ("vision_breastmnist", "breastmnist", 780, "Breast ultrasound tumor classification"),
        ("vision_retinamnist", "retinamnist", 1600, "Retinal fundus diabetic retinopathy classification"),
    ]

    for task_id, flag, max_n, desc in med_tasks:
        task_dir = OUTPUT_DIR / "vision" / task_id
        if (task_dir / "metadata.json").exists():
            logger.info(f"'{task_id}' already exists, skipping.")
            continue
        try:
            os.makedirs("/tmp/medmnist", exist_ok=True)
            DataClass = getattr(medmnist, INFO[flag]["python_class"])
            split_ds = DataClass(split="train", download=True, root="/tmp/medmnist")
            raw_imgs = split_ds.imgs
            raw_labels = split_ds.labels.squeeze()

            if len(raw_labels) > max_n:
                idx = np.random.RandomState(42).choice(len(raw_labels), max_n, replace=False)
                raw_imgs = raw_imgs[idx]
                raw_labels = raw_labels[idx]

            img_dir = RAW_IMG_DIR / task_id
            img_dir.mkdir(parents=True, exist_ok=True)
            img_paths = []
            for i, arr in enumerate(raw_imgs):
                p = img_dir / f"img_{i:05d}.png"
                if not p.exists():
                    Image.fromarray(arr).convert("RGB").save(p)
                img_paths.append(str(p))

            class_labels = list(INFO[flag]["label"].values())
            save_task("vision", task_id, img_paths, np.array(raw_labels, dtype=np.int64),
                       class_labels, f"MedMNIST ({flag})", desc, "json")
        except Exception as e:
            logger.error(f"FAILED vision task '{task_id}': {e}")


# ============================================================
# AUDIO: 3 new small real audio classification tasks
# ============================================================
def fetch_extra_audio():
    logger.info("=== Fetching extra audio tasks ===")
    from datasets import load_dataset
    import soundfile as sf

    # UrbanSound8K - 10 classes, environmental sounds
    task_id = "audio_urbansound8k"
    task_dir = OUTPUT_DIR / "audio" / task_id
    if (task_dir / "metadata.json").exists():
        logger.info(f"'{task_id}' already exists, skipping.")
    else:
        try:
            ds = load_dataset("danavery/urbansound8K", split="train")
            n = len(ds)
            target_n = min(n, 2000)
            idx = np.random.RandomState(42).choice(n, target_n, replace=False)
            aud_dir = RAW_AUD_DIR / task_id
            aud_dir.mkdir(parents=True, exist_ok=True)
            paths, labels = [], []
            class_names = None
            for i, ix in enumerate(idx):
                item = ds[int(ix)]
                p = aud_dir / f"sample_{i:04d}.wav"
                if not p.exists():
                    arr = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    sf.write(p, arr, sr)
                paths.append(str(p))
                labels.append(int(item["classID"]))
            class_names = [f"class_{c}" for c in sorted(set(labels))]
            save_task("audio", task_id, paths, np.array(labels, dtype=np.int64),
                       class_names, "HuggingFace (danavery/urbansound8K)",
                       "Urban environmental sound classification (10 classes)", "json")
        except Exception as e:
            logger.error(f"FAILED audio task '{task_id}': {e}")

    # EmoDB - 7 classes, German emotional speech
    task_id = "audio_emodb"
    task_dir = OUTPUT_DIR / "audio" / task_id
    if (task_dir / "metadata.json").exists():
        logger.info(f"'{task_id}' already exists, skipping.")
    else:
        try:
            ds = load_dataset("renumics/emodb", split="train")
            aud_dir = RAW_AUD_DIR / task_id
            aud_dir.mkdir(parents=True, exist_ok=True)
            paths, labels = [], []
            class_names = ds.features["emotion"].names
            for i, item in enumerate(ds):
                p = aud_dir / f"sample_{i:04d}.wav"
                if not p.exists():
                    arr = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    sf.write(p, arr, sr)
                paths.append(str(p))
                labels.append(int(item["emotion"]))
            save_task("audio", task_id, paths, np.array(labels, dtype=np.int64),
                       class_names, "HuggingFace (renumics/emodb)",
                       "German emotional speech classification (7 emotions)", "json")
        except Exception as e:
            logger.error(f"FAILED audio task '{task_id}': {e}")

    # CREMA-D - 6 classes, emotional speech
    task_id = "audio_crema_d"
    task_dir = OUTPUT_DIR / "audio" / task_id
    if (task_dir / "metadata.json").exists():
        logger.info(f"'{task_id}' already exists, skipping.")
    else:
        try:
            ds = load_dataset("confit/crema-d", split="train")
            n = len(ds)
            target_n = min(n, 2000)
            idx = np.random.RandomState(42).choice(n, target_n, replace=False)
            aud_dir = RAW_AUD_DIR / task_id
            aud_dir.mkdir(parents=True, exist_ok=True)
            paths, labels = [], []
            class_names = ds.features["label"].names
            for i, ix in enumerate(idx):
                item = ds[int(ix)]
                p = aud_dir / f"sample_{i:04d}.wav"
                if not p.exists():
                    arr = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    sf.write(p, arr, sr)
                paths.append(str(p))
                labels.append(int(item["label"]))
            save_task("audio", task_id, paths, np.array(labels, dtype=np.int64),
                       class_names, "HuggingFace (confit/crema-d)",
                       "Crowd-sourced emotional speech classification (6 emotions)", "json")
        except Exception as e:
            logger.error(f"FAILED audio task '{task_id}': {e}")


def main():
    fetch_extra_proteins()
    fetch_extra_vision()
    fetch_extra_audio()
    logger.info("Done fetching extra tasks.")


if __name__ == "__main__":
    main()
