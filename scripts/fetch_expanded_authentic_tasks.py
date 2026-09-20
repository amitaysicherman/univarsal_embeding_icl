#!/usr/bin/env python3
"""Expanded Authentic Data Fetcher for All 7 Modalities.

Expands benchmark to ~10 authentic classification tasks per domain:
- molecules: 24 TDC tasks (already existing)
- graphs: 10 tasks (MUTAG, BZR, COX2, PROTEINS + ENZYMES, NCI1, DD, PTC_MR, AIDS, DHFR)
- timeseries: 12 tasks (ECG200, ElectricDevices, FordA, GunPoint, ItalyPower, Wafer + TwoLeadECG, FordB, Chinatown, SonyAIBORobotSurface1, Strawberry, Plane)
- text: 11 tasks (AGNews, SST-2, TweetEmotion, TweetIrony, TweetSentiment, YelpPolarity + RottenTomatoes, IMDB, TweetOffensive, TweetHate, FinancialPhrasebank)
- vision: 10 tasks (CIFAR-10, Fashion-MNIST, SVHN + Beans, MNIST, DermaMNIST, BloodMNIST, PneumoniaMNIST, BreastMNIST, RetinaMNIST)
- proteins: 8 tasks (DeepLocBinary, DeepLocMulti, MetalIon, SAbDabChen + Solubility, IEDB_Jespersen)
- audio: 4-5 tasks (ESC-10, ESC-50 + GTZAN Genre, Speech Commands 10)

Strict criteria:
- 2 to 10 classes
- <= 10k samples per task
- Authentic splits / 5-fold CV
- Real data files on disk (PNG, WAV, PyG, AEON, SMILES, Amino Acids)
- ZERO synthetic fallbacks.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_expanded")

OUTPUT_DIR = Path("data/tasks")
RAW_IMG_DIR = Path("data/raw/images")
RAW_AUD_DIR = Path("data/raw/audio")


def save_task(
    domain: str,
    task_id: str,
    inputs: list | np.ndarray,
    labels: np.ndarray,
    class_names: list,
    source: str,
    description: str,
    folds: Optional[np.ndarray] = None,
    input_format: str = "json",
) -> None:
    task_dir = OUTPUT_DIR / domain / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int64)
    unique_classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)

    # Remap classes to 0..C-1 if needed
    if not np.array_equal(unique_classes, np.arange(num_classes)):
        remap = {c: i for i, c in enumerate(unique_classes)}
        labels = np.array([remap[c] for c in labels], dtype=np.int64)
        unique_classes, counts = np.unique(labels, return_counts=True)

    num_samples = len(labels)

    # Generate 5 folds if not provided
    if folds is None or len(folds) != num_samples:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = np.zeros(num_samples, dtype=np.int64)
        for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(num_samples), labels)):
            folds[test_idx] = fold_idx

    # Save inputs
    if input_format == "json":
        with open(task_dir / "inputs.json", "w") as f:
            json.dump(inputs, f)
    elif input_format == "npy":
        np.save(task_dir / "inputs.npy", np.asarray(inputs, dtype=np.float32))
    elif input_format == "pt":
        import torch
        torch.save(inputs, task_dir / "inputs.pt")

    # Save labels and folds
    np.save(task_dir / "labels.npy", labels)
    np.save(task_dir / "folds.npy", folds)

    # Save split_indices (Fold 0 as train/test)
    train_idx = np.where(folds != 0)[0].tolist()
    test_idx = np.where(folds == 0)[0].tolist()
    with open(task_dir / "split_indices.json", "w") as f:
        json.dump({"train": train_idx, "val": [], "test": test_idx}, f, indent=2)

    # Save metadata
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

    logger.info(f"Saved [{domain.upper()}] {task_id}: N={num_samples}, C={num_classes}, classes={dict(zip(unique_classes.tolist(), counts.tolist()))}")


# ==============================================================================
# 1. GRAPHS (PyG TUDataset)
# ==============================================================================
def fetch_expanded_graphs():
    logger.info("=== Fetching Expanded Graph Tasks ===")
    from torch_geometric.datasets import TUDataset

    tasks = [
        ("graph_enzymes", "ENZYMES", "Protein enzymes classified into 6 EC levels"),
        ("graph_nci1", "NCI1", "Chemical compounds screened against lung cancer"),
        ("graph_dd", "DD", "Enzymes vs non-enzymes protein structures"),
        ("graph_ptc_mr", "PTC_MR", "Carcinogenicity in male rats"),
        ("graph_aids", "AIDS", "Anti-HIV activity screening"),
        ("graph_dhfr", "DHFR", "Dihydrofolate reductase inhibitors"),
    ]

    for task_id, tu_name, desc in tasks:
        task_dir = OUTPUT_DIR / "graphs" / task_id
        if (task_dir / "metadata.json").exists() and (task_dir / "inputs.pt").exists():
            logger.info(f"Task '{task_id}' already exists. Skipping.")
            continue

        try:
            ds = TUDataset(root=f"/tmp/tudataset_{tu_name}", name=tu_name)
            labels = np.array([int(data.y.item()) for data in ds])
            graphs_list = [data for data in ds]

            # Subsample if > 10k
            if len(labels) > 10000:
                indices = np.random.RandomState(42).choice(len(labels), 10000, replace=False)
                graphs_list = [graphs_list[i] for i in indices]
                labels = labels[indices]

            save_task(
                domain="graphs",
                task_id=task_id,
                inputs=graphs_list,
                labels=labels,
                class_names=[f"class_{i}" for i in range(len(np.unique(labels)))],
                source=f"PyG TUDataset ({tu_name})",
                description=desc,
                input_format="pt",
            )
        except Exception as e:
            logger.error(f"Failed to fetch graph task '{task_id}': {e}")


# ==============================================================================
# 2. TIME SERIES (AEON / UCR Archive)
# ==============================================================================
def fetch_expanded_timeseries():
    logger.info("=== Fetching Expanded Time Series Tasks ===")
    from aeon.datasets import load_classification

    tasks = [
        ("ts_two_lead_ecg", "TwoLeadECG", "Two-lead ECG heartbeat classification"),
        ("ts_ford_b", "FordB", "Engine noise symptom classification under noisy condition"),
        ("ts_chinatown", "Chinatown", "Pedestrian traffic flow in Melbourne Chinatown"),
        ("ts_sony_aibo", "SonyAIBORobotSurface1", "Robot accelerometer surface classification"),
        ("ts_strawberry", "Strawberry", "Strawberry puree food spectrography authentication"),
        ("ts_plane", "Plane", "Airplane shape contour time series classification"),
    ]

    for task_id, ucr_name, desc in tasks:
        task_dir = OUTPUT_DIR / "timeseries" / task_id
        if (task_dir / "metadata.json").exists() and (task_dir / "inputs.npy").exists():
            logger.info(f"Task '{task_id}' already exists. Skipping.")
            continue

        try:
            X, y = load_classification(ucr_name)
            if X.ndim == 3 and X.shape[1] == 1:
                X = X.squeeze(1)

            # Map labels to integers
            u_labels = sorted(list(set(y)))
            label_map = {lab: idx for idx, lab in enumerate(u_labels)}
            int_labels = np.array([label_map[lab] for lab in y], dtype=np.int64)

            # Subsample if > 10k
            if len(int_labels) > 10000:
                indices = np.random.RandomState(42).choice(len(int_labels), 10000, replace=False)
                X = X[indices]
                int_labels = int_labels[indices]

            save_task(
                domain="timeseries",
                task_id=task_id,
                inputs=X,
                labels=int_labels,
                class_names=[str(l) for l in u_labels],
                source=f"UCR Archive ({ucr_name}) via AEON",
                description=desc,
                input_format="npy",
            )
        except Exception as e:
            logger.error(f"Failed to fetch time series task '{task_id}': {e}")


# ==============================================================================
# 3. TEXT (Hugging Face Datasets)
# ==============================================================================
def fetch_expanded_text():
    logger.info("=== Fetching Expanded Text Tasks ===")
    from datasets import load_dataset

    tasks = [
        ("text_rotten_tomatoes", "rotten_tomatoes", None, "text", "label", ["negative", "positive"], "Rotten Tomatoes movie reviews sentiment"),
        ("text_imdb", "imdb", None, "text", "label", ["negative", "positive"], "IMDb large movie reviews sentiment"),
        ("text_tweet_offensive", "tweet_eval", "offensive", "text", "label", ["not_offensive", "offensive"], "Offensive language detection in tweets"),
        ("text_tweet_hate", "tweet_eval", "hate", "text", "label", ["not_hate", "hate"], "Hate speech detection in tweets"),
        ("text_financial_phrasebank", "financial_phrasebank", "sentences_allagree", "sentence", "label", ["negative", "neutral", "positive"], "Financial sentiment classification"),
    ]

    for task_id, dset_name, sub, text_col, label_col, class_names, desc in tasks:
        task_dir = OUTPUT_DIR / "text" / task_id
        if (task_dir / "metadata.json").exists() and (task_dir / "inputs.json").exists():
            logger.info(f"Task '{task_id}' already exists. Skipping.")
            continue

        try:
            ds = load_dataset(dset_name, sub, split="train") if sub else load_dataset(dset_name, split="train")
            texts = [str(x).strip() for x in ds[text_col]]
            labels = np.array(ds[label_col], dtype=np.int64)

            # Filter valid
            valid_idx = [i for i, t in enumerate(texts) if len(t) > 0 and labels[i] >= 0]
            texts = [texts[i] for i in valid_idx]
            labels = labels[valid_idx]

            # Subsample to <= 3000 for efficiency
            target_n = min(len(labels), 3000)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for _, keep_idx in skf.split(np.zeros(len(labels)), labels):
                if len(keep_idx) >= target_n:
                    indices = keep_idx[:target_n]
                    break
            else:
                indices = np.random.RandomState(42).choice(len(labels), target_n, replace=False)

            texts = [texts[i] for i in indices]
            labels = labels[indices]

            save_task(
                domain="text",
                task_id=task_id,
                inputs=texts,
                labels=labels,
                class_names=class_names,
                source=f"HuggingFace ({dset_name}/{sub or ''})",
                description=desc,
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch text task '{task_id}': {e}")


# ==============================================================================
# 4. VISION (MedMNIST & Real Images)
# ==============================================================================
def fetch_expanded_vision():
    logger.info("=== Fetching Expanded Vision Tasks ===")
    from PIL import Image
    from datasets import load_dataset
    import medmnist
    from medmnist import INFO

    # 4a. Beans (real leaf photos)
    beans_id = "vision_beans"
    beans_dir = OUTPUT_DIR / "vision" / beans_id
    if not (beans_dir / "metadata.json").exists():
        try:
            ds = load_dataset("beans", split="train")
            img_dir = RAW_IMG_DIR / beans_id
            img_dir.mkdir(parents=True, exist_ok=True)

            img_paths = []
            labels = []
            for idx, item in enumerate(ds):
                p = img_dir / f"img_{idx:05d}.png"
                if not p.exists():
                    img = item["image"].convert("RGB")
                    img.save(p)
                img_paths.append(str(p))
                labels.append(int(item["labels"]))

            save_task(
                domain="vision",
                task_id=beans_id,
                inputs=img_paths,
                labels=np.array(labels, dtype=np.int64),
                class_names=["angular_leaf_spot", "bean_rust", "healthy"],
                source="HuggingFace (beans)",
                description="Bean leaf disease classification",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch vision beans: {e}")

    # 4b. MNIST (10 classes digits, 1000 samples)
    mnist_id = "vision_mnist"
    mnist_dir = OUTPUT_DIR / "vision" / mnist_id
    if not (mnist_dir / "metadata.json").exists():
        try:
            ds = load_dataset("mnist", split="test")
            img_dir = RAW_IMG_DIR / mnist_id
            img_dir.mkdir(parents=True, exist_ok=True)

            # Sample 1000
            indices = np.random.RandomState(42).choice(len(ds), 1000, replace=False)
            img_paths = []
            labels = []
            for idx in indices:
                item = ds[int(idx)]
                p = img_dir / f"img_{idx:05d}.png"
                if not p.exists():
                    img = item["image"].convert("RGB")
                    img.save(p)
                img_paths.append(str(p))
                labels.append(int(item["label"]))

            save_task(
                domain="vision",
                task_id=mnist_id,
                inputs=img_paths,
                labels=np.array(labels, dtype=np.int64),
                class_names=[str(i) for i in range(10)],
                source="HuggingFace (mnist)",
                description="Handwritten digit classification (10 classes)",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch vision mnist: {e}")

    # 4c. MedMNIST tasks
    med_tasks = [
        ("vision_dermamnist", "dermamnist", 2000, "Dermatoscopy skin lesion classification (7 classes)"),
        ("vision_bloodmnist", "bloodmnist", 2000, "Peripheral blood cell classification (8 classes)"),
        ("vision_pneumoniamnist", "pneumoniamnist", 2000, "Pediatric chest X-ray pneumonia classification (2 classes)"),
        ("vision_breastmnist", "breastmnist", 780, "Breast ultrasound tumor classification (2 classes)"),
        ("vision_retinamnist", "retinamnist", 1600, "Retinal fundus diabetic retinopathy classification (5 classes)"),
    ]

    for task_id, flag, max_n, desc in med_tasks:
        task_dir = OUTPUT_DIR / "vision" / task_id
        if (task_dir / "metadata.json").exists():
            logger.info(f"Task '{task_id}' already exists. Skipping.")
            continue

        try:
            os.makedirs("/tmp/medmnist", exist_ok=True)
            DataClass = getattr(medmnist, INFO[flag]["python_class"])
            split_ds = DataClass(split="train", download=True, root="/tmp/medmnist")
            raw_imgs = split_ds.imgs  # (N, 28, 28) or (N, 28, 28, 3)
            raw_labels = split_ds.labels.squeeze()

            if len(raw_labels) > max_n:
                indices = np.random.RandomState(42).choice(len(raw_labels), max_n, replace=False)
                raw_imgs = raw_imgs[indices]
                raw_labels = raw_labels[indices]

            img_dir = RAW_IMG_DIR / task_id
            img_dir.mkdir(parents=True, exist_ok=True)

            img_paths = []
            for i, arr in enumerate(raw_imgs):
                p = img_dir / f"img_{i:05d}.png"
                if not p.exists():
                    img = Image.fromarray(arr).convert("RGB")
                    img.save(p)
                img_paths.append(str(p))

            class_labels = list(INFO[flag]["label"].values())
            save_task(
                domain="vision",
                task_id=task_id,
                inputs=img_paths,
                labels=np.array(raw_labels, dtype=np.int64),
                class_names=class_labels,
                source=f"MedMNIST ({flag})",
                description=desc,
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch vision medmnist task '{task_id}': {e}")


# ==============================================================================
# 5. PROTEINS (SaProtHub & TDC)
# ==============================================================================
def fetch_expanded_proteins():
    logger.info("=== Fetching Expanded Protein Tasks ===")
    from datasets import load_dataset
    from tdc.single_pred import Epitope

    # Fix sabdab_chen if needed
    sabdab_dir = OUTPUT_DIR / "proteins" / "prot_sabdab_chen"
    if (sabdab_dir / "inputs.json").exists():
        with open(sabdab_dir / "inputs.json") as f:
            raw = json.load(f)
        if len(raw) > 0 and raw[0].startswith("['"):
            logger.info("Fixing SAbDab Chen raw chain sequence strings...")
            cleaned = []
            for s in raw:
                try:
                    chains = ast.literal_eval(s)
                    clean_seq = "".join(chains)
                except Exception:
                    clean_seq = "".join(c for c in s if c.isalpha()).upper()
                cleaned.append(clean_seq)
            with open(sabdab_dir / "inputs.json", "w") as f:
                json.dump(cleaned, f)
            logger.info(f"Fixed {len(cleaned)} SAbDab sequences.")

    # 5a. SaProtHub Solubility
    sol_id = "prot_solubility"
    sol_dir = OUTPUT_DIR / "proteins" / sol_id
    if not (sol_dir / "metadata.json").exists():
        try:
            ds = load_dataset("SaProtHub/Dataset-Solubility", split="train")
            proteins = [str(p).strip().upper() for p in ds["protein"]]
            labels = np.array(ds["label"], dtype=np.int64)

            # Subsample to 2500
            indices = np.random.RandomState(42).choice(len(labels), 2500, replace=False)
            proteins = [proteins[i] for i in indices]
            labels = labels[indices]

            save_task(
                domain="proteins",
                task_id=sol_id,
                inputs=proteins,
                labels=labels,
                class_names=["insoluble", "soluble"],
                source="SaProtHub (Dataset-Solubility)",
                description="Protein solubility classification (binary)",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch protein solubility: {e}")

    # 5b. Proteinea Remote Homology
    rh_id = "prot_remote_homology"
    rh_dir = OUTPUT_DIR / "proteins" / rh_id
    if not (rh_dir / "metadata.json").exists():
        try:
            ds = load_dataset("proteinea/remote_homology", split="train")
            proteins = [str(p).strip().upper() for p in ds["primary"]]
            labels = np.array(ds["class_label"], dtype=np.int64)

            target_n = min(len(labels), 3000)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for _, keep_idx in skf.split(np.zeros(len(labels)), labels):
                if len(keep_idx) >= target_n:
                    indices = keep_idx[:target_n]
                    break
            else:
                indices = np.random.RandomState(42).choice(len(labels), target_n, replace=False)

            proteins = [proteins[i] for i in indices]
            labels = labels[indices]

            save_task(
                domain="proteins",
                task_id=rh_id,
                inputs=proteins,
                labels=labels,
                class_names=["all_alpha", "all_beta", "alpha_beta", "alpha_plus_beta", "multi_domain", "membrane", "small_protein"],
                source="HuggingFace (proteinea/remote_homology)",
                description="Protein fold structural classification (7 classes)",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch protein remote_homology: {e}")


# ==============================================================================
# 6. AUDIO (GTZAN Genre & SUPERB Keyword Spotting)
# ==============================================================================
def fetch_expanded_audio():
    logger.info("=== Fetching Expanded Audio Tasks ===")
    from datasets import load_dataset
    import soundfile as sf

    gtzan_id = "audio_gtzan"
    gtzan_dir = OUTPUT_DIR / "audio" / gtzan_id
    if not (gtzan_dir / "metadata.json").exists():
        try:
            ds = load_dataset("marsyas/gtzan", split="train")
            aud_dir = RAW_AUD_DIR / gtzan_id
            aud_dir.mkdir(parents=True, exist_ok=True)

            wav_paths = []
            labels = []
            class_names = ds.features["genre"].names

            for idx, item in enumerate(ds):
                p = aud_dir / f"sample_{idx:04d}.wav"
                if not p.exists():
                    audio_arr = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    sf.write(p, audio_arr, sr)
                wav_paths.append(str(p))
                labels.append(int(item["genre"]))

            save_task(
                domain="audio",
                task_id=gtzan_id,
                inputs=wav_paths,
                labels=np.array(labels, dtype=np.int64),
                class_names=class_names,
                source="HuggingFace (marsyas/gtzan)",
                description="Music genre classification across 10 genres (GTZAN)",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch audio gtzan: {e}")

    ks_id = "audio_superb_ks"
    ks_dir = OUTPUT_DIR / "audio" / ks_id
    if not (ks_dir / "metadata.json").exists():
        try:
            ds = load_dataset("superb", "ks", split="train")
            aud_dir = RAW_AUD_DIR / ks_id
            aud_dir.mkdir(parents=True, exist_ok=True)

            indices = np.random.RandomState(42).choice(len(ds), 2000, replace=False)
            wav_paths = []
            labels = []

            for i, idx in enumerate(indices):
                item = ds[int(idx)]
                p = aud_dir / f"sample_{i:04d}.wav"
                if not p.exists():
                    audio_arr = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    sf.write(p, audio_arr, sr)
                wav_paths.append(str(p))
                labels.append(int(item["label"]))

            save_task(
                domain="audio",
                task_id=ks_id,
                inputs=wav_paths,
                labels=np.array(labels, dtype=np.int64),
                class_names=[f"keyword_{i}" for i in range(12)],
                source="HuggingFace (superb/ks)",
                description="Keyword spotting speech recognition (10 core words)",
                input_format="json",
            )
        except Exception as e:
            logger.error(f"Failed to fetch audio superb_ks: {e}")


def main():
    logger.info("Starting acquisition of expanded authentic tasks...")
    fetch_expanded_graphs()
    fetch_expanded_timeseries()
    fetch_expanded_text()
    fetch_expanded_vision()
    fetch_expanded_proteins()
    fetch_expanded_audio()
    logger.info("Expanded authentic data acquisition finished!")


if __name__ == "__main__":
    main()
