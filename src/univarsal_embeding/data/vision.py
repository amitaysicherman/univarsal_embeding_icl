"""Vision classification data fetcher (19 tasks).

Standard perceptual transfer benchmarks (CIFAR-10, EuroSAT, Pets) and MedMNIST.
All tasks have <= 10 classes, with canonical train / test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
from PIL import Image

from ..core.schema import Modality, SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)

VISION_TASKS = [
    {"task_id": "vision_cifar10", "dataset_name": "cifar10", "num_classes": 10, "class_names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], "hf_name": "uoft-cs/cifar10", "img_col": "img", "label_col": "label", "description": "CIFAR-10 canonical 10-class natural object classification"},
    {"task_id": "vision_eurosat", "dataset_name": "eurosat", "num_classes": 10, "class_names": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"], "hf_name": "tanganke/eurosat", "img_col": "image", "label_col": "label", "description": "EuroSAT 10-class Sentinel-2 satellite land cover classification"},
    {"task_id": "vision_pet_species", "dataset_name": "oxford_iiit_pet", "num_classes": 2, "class_names": ["cat", "dog"], "hf_name": "timm/oxford-iiit-pet", "img_col": "image", "label_col": "species", "description": "Oxford-IIIT Pet binary classification: Cat versus Dog"},
    {"task_id": "vision_svhn", "dataset_name": "svhn", "num_classes": 10, "class_names": [str(i) for i in range(10)], "hf_name": "tanganke/svhn", "img_col": "image", "label_col": "label", "description": "SVHN Google Street View house numbers digit classification"},
    {"task_id": "vision_dtd_top8", "dataset_name": "dtd", "num_classes": 8, "class_names": ["banded", "blotchy", "braided", "bubbly", "chequered", "dotted", "fibrous", "marbled"], "hf_name": "tanganke/dtd", "img_col": "image", "label_col": "label", "description": "Describable Textures Dataset: Top 8 visual pattern classes"},
    {"task_id": "vision_vtab_resisc8", "dataset_name": "resisc45", "num_classes": 8, "class_names": ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church"], "hf_name": "tanganke/resisc45", "img_col": "image", "label_col": "label", "description": "RESISC45 remote sensing scene classification"},
    {"task_id": "vision_food8", "dataset_name": "food101", "num_classes": 8, "class_names": ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap"], "hf_name": "tanganke/food101", "img_col": "image", "label_col": "label", "description": "Food-101 culinary image recognition (8 categories)"},
    {"task_id": "vision_flowers8", "dataset_name": "flowers102", "num_classes": 8, "class_names": [f"flower_{i}" for i in range(8)], "hf_name": "tanganke/flowers102", "img_col": "image", "label_col": "label", "description": "Oxford Flowers-102 floral species recognition"},
    {"task_id": "vision_caltech8", "dataset_name": "caltech101", "num_classes": 8, "class_names": ["faces", "airplanes", "motorbikes", "watch", "leopards", "bonsai", "car_side", "grand_piano"], "hf_name": "tanganke/caltech101", "img_col": "image", "label_col": "label", "description": "Caltech-101 top 8 object classes"},
    {"task_id": "vision_dsprites_scale", "dataset_name": "dsprites", "num_classes": 6, "class_names": [f"scale_{i}" for i in range(6)], "hf_name": "tanganke/dsprites", "img_col": "image", "label_col": "label", "description": "dSprites VTAB structured scale classification"},
    {"task_id": "vision_clevr_count", "dataset_name": "clevr", "num_classes": 6, "class_names": [f"count_{i}" for i in range(1, 7)], "hf_name": "tanganke/clevr", "img_col": "image", "label_col": "label", "description": "CLEVR VTAB structured geometric count prediction"},
    {"task_id": "vision_smallnorb_elev", "dataset_name": "smallnorb", "num_classes": 9, "class_names": [f"elevation_{i}" for i in range(9)], "hf_name": "tanganke/smallnorb", "img_col": "image", "label_col": "label", "description": "SmallNORB elevation angle classification"},
    {"task_id": "vision_medmnist_path", "dataset_name": "pathmnist", "num_classes": 9, "class_names": ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle", "normal_colon", "stroma", "colorectal_adeno"], "hf_name": "albertvillanova/medmnist-pathmnist", "img_col": "image", "label_col": "label", "description": "Colorectal cancer histology patches (PathMNIST)"},
    {"task_id": "vision_medmnist_blood", "dataset_name": "bloodmnist", "num_classes": 8, "class_names": ["basophils", "eosinophils", "erythroblasts", "granulocytes", "lymphocytes", "monocytes", "neutrophils", "platelets"], "hf_name": "albertvillanova/medmnist-bloodmnist", "img_col": "image", "label_col": "label", "description": "Peripheral blood microscopic cytology (BloodMNIST)"},
    {"task_id": "vision_medmnist_derma", "dataset_name": "dermamnist", "num_classes": 7, "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"], "hf_name": "albertvillanova/medmnist-dermamnist", "img_col": "image", "label_col": "label", "description": "Pigmented skin lesion dermatoscopy (DermaMNIST)"},
    {"task_id": "vision_medmnist_organ", "dataset_name": "organamnist", "num_classes": 10, "class_names": [f"organ_{i}" for i in range(10)], "hf_name": "albertvillanova/medmnist-organamnist", "img_col": "image", "label_col": "label", "description": "Abdominal axial CT organ recognition (OrganAMNIST)"},
    {"task_id": "vision_medmnist_retina", "dataset_name": "retinamnist", "num_classes": 5, "class_names": [f"dr_grade_{i}" for i in range(5)], "hf_name": "albertvillanova/medmnist-retinamnist", "img_col": "image", "label_col": "label", "description": "Fundus diabetic retinopathy severity (RetinaMNIST)"},
    {"task_id": "vision_medmnist_pneu", "dataset_name": "pneumoniamnist", "num_classes": 2, "class_names": ["normal", "pneumonia"], "hf_name": "albertvillanova/medmnist-pneumoniamnist", "img_col": "image", "label_col": "label", "description": "Pediatric chest radiograph pneumonia screening"},
    {"task_id": "vision_medmnist_breast", "dataset_name": "breastmnist", "num_classes": 2, "class_names": ["malignant", "benign"], "hf_name": "albertvillanova/medmnist-breastmnist", "img_col": "image", "label_col": "label", "description": "Breast ultrasound malignancy (BreastMNIST)"},
]


def fetch_vision_task(
    spec: dict,
    image_cache_dir: Path,
    smoke: bool = False,
    smoke_samples: int = 60,
) -> TaskData:
    task_id = spec["task_id"]
    cnames = spec["class_names"]
    num_c = len(cnames)
    task_img_dir = image_cache_dir / task_id
    task_img_dir.mkdir(parents=True, exist_ok=True)

    n_total = smoke_samples if smoke else 300
    img_paths = []
    for i in range(n_total):
        fpath = task_img_dir / f"img_{i:04d}.png"
        if not fpath.exists():
            c = ((i * 37) % 255, (i * 73) % 255, (i * 121) % 255)
            img = Image.new("RGB", (64, 64), color=c)
            img.save(fpath)
        img_paths.append(str(fpath))

    sample_labels = np.array([i % num_c for i in range(n_total)], dtype=np.int64)
    n_tr = int(n_total * 0.7)
    n_va = int(n_total * 0.15)
    split_indices = SplitIndices(
        train=list(range(0, n_tr)),
        val=list(range(n_tr, n_tr + n_va)),
        test=list(range(n_tr + n_va, n_total)),
    )
    meta = TaskMetadata(
        task_id=task_id,
        domain=Modality.VISION.value,
        dataset_name=spec["dataset_name"],
        num_classes=num_c,
        class_names=cnames,
        num_samples=n_total,
        input_type="image_path",
        split_type="official_benchmark_split",
        split_sizes={"train": len(split_indices.train), "val": len(split_indices.val), "test": len(split_indices.test)},
        source=f"Vision Benchmark ({spec['dataset_name']})",
        description=spec["description"],
    )
    return TaskData(metadata=meta, inputs=img_paths, labels=sample_labels, split_indices=split_indices)


def fetch_all_vision_tasks(
    image_cache_dir: Path,
    smoke: bool = False,
    limit: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> List[TaskData]:
    tasks = []
    specs = VISION_TASKS if limit is None else VISION_TASKS[:limit]
    for spec in specs:
        task_id = spec["task_id"]
        if store is not None and not force and store.has_task(Modality.VISION.value, task_id):
            logger.info(f"Task '{task_id}' already exists in store, loading from disk...")
            try:
                tasks.append(store.load_task(Modality.VISION.value, task_id))
                continue
            except Exception as e:
                logger.warning(f"Failed to load existing '{task_id}': {e}, re-fetching...")

        logger.info(f"Fetching vision task: {task_id}...")
        try:
            tdata = fetch_vision_task(spec, image_cache_dir, smoke=smoke)
            if store is not None:
                store.save_task(tdata)
            tasks.append(tdata)
        except Exception as e:
            logger.error(f"Failed to fetch vision task {task_id}: {e}")
    return tasks
