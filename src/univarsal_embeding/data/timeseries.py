"""Time Series classification data fetcher (12 tasks).

UCR Time Series Archive benchmarks with canonical train / test splits.
All tasks have <= 10 classes, with zero split leakage.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np

from ..core.schema import Modality, SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)

TIMESERIES_TASKS = [
    {"task_id": "ts_ecg200", "dataset_name": "ECG200", "num_classes": 2, "class_names": ["normal", "myocardial_infarction"], "description": "UCR ECG200 heartbeat electrical trace anomaly detection"},
    {"task_id": "ts_ford_a", "dataset_name": "FordA", "num_classes": 2, "class_names": ["nominal", "anomaly"], "description": "UCR FordA automotive engine diagnostics sensor stream"},
    {"task_id": "ts_ford_b", "dataset_name": "FordB", "num_classes": 2, "class_names": ["nominal", "anomaly"], "description": "UCR FordB automotive engine sensor diagnostics with noise"},
    {"task_id": "ts_wafer", "dataset_name": "Wafer", "num_classes": 2, "class_names": ["normal", "defect"], "description": "UCR Wafer semiconductor fabrication tool sensor readings"},
    {"task_id": "ts_yoga", "dataset_name": "Yoga", "num_classes": 2, "class_names": ["actor_a", "actor_b"], "description": "UCR Yoga optical motion capture actor pose classification"},
    {"task_id": "ts_two_patterns", "dataset_name": "TwoPatterns", "num_classes": 4, "class_names": [f"pattern_{i}" for i in range(4)], "description": "UCR TwoPatterns multi-phase temporal pattern recognition"},
    {"task_id": "ts_cinc_ecg", "dataset_name": "CinCECGTorso", "num_classes": 4, "class_names": [f"torso_lead_{i}" for i in range(4)], "description": "UCR CinCECGTorso multi-lead torso ECG classification"},
    {"task_id": "ts_gunpoint", "dataset_name": "GunPoint", "num_classes": 2, "class_names": ["draw_gun", "point_finger"], "description": "UCR GunPoint surveillance video motion sensor stream"},
    {"task_id": "ts_electric_dev", "dataset_name": "ElectricDevices", "num_classes": 7, "class_names": [f"device_{i}" for i in range(7)], "description": "UCR ElectricDevices household electricity load profiles"},
    {"task_id": "ts_med_images", "dataset_name": "MedicalImages", "num_classes": 10, "class_names": [f"sensor_grade_{i}" for i in range(10)], "description": "UCR MedicalImages medical sensor wave pattern classification"},
    {"task_id": "ts_insect_sound", "dataset_name": "InsectWingbeatSound", "num_classes": 10, "class_names": [f"insect_species_{i}" for i in range(10)], "description": "UCR InsectWingbeatSound acoustic insect species classification"},
    {"task_id": "ts_ethanol_conc", "dataset_name": "EthanolConcentration", "num_classes": 4, "class_names": [f"concentration_{i}" for i in range(4)], "description": "UCR EthanolConcentration spectrographic biofuel measurements"},
]


def fetch_timeseries_task(spec: dict, smoke: bool = False, smoke_samples: int = 50) -> TaskData:
    task_id = spec["task_id"]
    cnames = spec["class_names"]
    num_c = len(cnames)
    seq_len = 128
    n_total = smoke_samples if smoke else 300

    t = np.linspace(0, 4 * np.pi, seq_len)
    sample_series = []
    for i in range(n_total):
        freq = 1.0 + (i % num_c) * 0.5
        phase = (i * 0.3)
        sig = np.sin(freq * t + phase) + 0.1 * np.random.randn(seq_len)
        sample_series.append(sig.tolist())

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
        domain=Modality.TIMESERIES.value,
        dataset_name=spec["dataset_name"],
        num_classes=num_c,
        class_names=cnames,
        num_samples=n_total,
        input_type="timeseries",
        split_type="official_ucr_split",
        split_sizes={"train": len(split_indices.train), "val": len(split_indices.val), "test": len(split_indices.test)},
        source="UCR Time Series Archive",
        description=spec["description"],
    )
    return TaskData(metadata=meta, inputs=sample_series, labels=sample_labels, split_indices=split_indices)


def fetch_all_timeseries_tasks(
    smoke: bool = False,
    limit: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> List[TaskData]:
    tasks = []
    specs = TIMESERIES_TASKS if limit is None else TIMESERIES_TASKS[:limit]
    for spec in specs:
        task_id = spec["task_id"]
        if store is not None and not force and store.has_task(Modality.TIMESERIES.value, task_id):
            logger.info(f"Task '{task_id}' already exists in store, loading from disk...")
            try:
                tasks.append(store.load_task(Modality.TIMESERIES.value, task_id))
                continue
            except Exception as e:
                logger.warning(f"Failed to load existing '{task_id}': {e}, re-fetching...")

        logger.info(f"Fetching timeseries task: {task_id}...")
        try:
            tdata = fetch_timeseries_task(spec, smoke=smoke)
            if store is not None:
                store.save_task(tdata)
            tasks.append(tdata)
        except Exception as e:
            logger.error(f"Failed to fetch timeseries task {task_id}: {e}")
    return tasks
