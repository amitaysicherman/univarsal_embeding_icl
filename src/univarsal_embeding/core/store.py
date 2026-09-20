"""Standalone storage manager for task inputs, labels, canonical splits, and embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from .schema import SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)


class TaskStore:
    """Manages reading and writing task data and representation embeddings."""

    def __init__(self, data_root: Path | str) -> None:
        self.data_root = Path(data_root)
        self.tasks_root = self.data_root / "tasks"
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, domain: str, task_id: str) -> Path:
        return self.tasks_root / domain / task_id

    def has_task(self, domain: str, task_id: str) -> bool:
        """Returns True if task metadata, inputs, and labels already exist on disk."""
        tdir = self.task_dir(domain, task_id)
        return (
            (tdir / "metadata.json").exists()
            and (tdir / "inputs.json").exists()
            and (tdir / "labels.npy").exists()
            and (tdir / "split_indices.json").exists()
        )

    def save_task(self, task_data: TaskData) -> Path:
        """Persists a complete task with inputs, labels, metadata, and canonical splits."""
        tdir = self.task_dir(task_data.metadata.domain, task_data.metadata.task_id)
        tdir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata
        meta_path = tdir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(task_data.metadata.to_dict(), f, indent=2)

        # 2. Save inputs
        inputs_path = tdir / "inputs.json"
        with open(inputs_path, "w", encoding="utf-8") as f:
            json.dump(task_data.inputs, f)

        # 3. Save labels
        labels_path = tdir / "labels.npy"
        np.save(labels_path, task_data.labels.astype(np.int64))

        # 4. Save explicit split indices
        split_path = tdir / "split_indices.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(task_data.split_indices.to_dict(), f, indent=2)

        # 5. Save canonical folds (0=test, 1=val, 2=train) for instant ICL compatibility
        folds_path = tdir / "folds.npy"
        folds = task_data.split_indices.to_folds_array(len(task_data.labels))
        np.save(folds_path, folds)

        logger.info(
            f"Saved task '{task_data.metadata.task_id}' ({task_data.metadata.domain}): "
            f"{len(task_data.labels)} samples, classes={task_data.metadata.num_classes}, "
            f"splits={task_data.metadata.split_sizes}"
        )
        return tdir

    def load_task(self, domain: str, task_id: str) -> TaskData:
        """Loads a task completely into memory."""
        tdir = self.task_dir(domain, task_id)
        if not tdir.exists():
            raise FileNotFoundError(f"Task '{task_id}' not found in {tdir}")

        with open(tdir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = TaskMetadata.from_dict(json.load(f))

        if (tdir / "inputs.json").exists():
            with open(tdir / "inputs.json", "r", encoding="utf-8") as f:
                inputs = json.load(f)
        elif (tdir / "inputs.npy").exists():
            inputs = np.load(tdir / "inputs.npy", allow_pickle=True)
        elif (tdir / "inputs.pt").exists():
            import torch
            inputs = torch.load(tdir / "inputs.pt", weights_only=False)
        else:
            raise FileNotFoundError(
                f"No inputs file (inputs.json/.npy/.pt) found in {tdir}"
            )

        labels = np.load(tdir / "labels.npy")

        if (tdir / "folds.npy").exists():
            folds = np.load(tdir / "folds.npy")
            # Convention: fold 0 = test, fold 1 = val, all remaining folds (2, 3, 4, ...)
            # = train. This covers both the plain 3-way (0/1/2) encoding and the
            # k-fold CV encoding (0..k-1) used by some tasks without dropping samples.
            test_idx = np.flatnonzero(folds == 0).tolist()
            val_idx = np.flatnonzero(folds == 1).tolist()
            train_idx = np.flatnonzero(folds >= 2).tolist()
            split_indices = SplitIndices(train=train_idx, val=val_idx, test=test_idx)
        elif (tdir / "split_indices.json").exists():
            with open(tdir / "split_indices.json", "r", encoding="utf-8") as f:
                split_indices = SplitIndices.from_dict(json.load(f))
        else:
            # Fallback 70/15/15 split if no split files exist
            n = len(labels)
            n_te = max(1, int(n * 0.15))
            n_va = max(1, int(n * 0.15))
            split_indices = SplitIndices(
                train=list(range(n_te + n_va, n)),
                val=list(range(n_te, n_te + n_va)),
                test=list(range(n_te)),
            )

        return TaskData(metadata=metadata, inputs=inputs, labels=labels, split_indices=split_indices)

    def save_embeddings(self, domain: str, task_id: str, encoder_id: str, embeddings: np.ndarray) -> Path:
        """Saves dense representation embeddings matrix (N, d) for a specific encoder."""
        tdir = self.task_dir(domain, task_id)
        emb_dir = tdir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / f"{encoder_id}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        logger.info(f"Saved embeddings '{encoder_id}' {embeddings.shape} to {emb_path}")
        return emb_path

    def load_embeddings(
        self, domain: str, task_id: str, encoder_id: str, mmap_mode: Optional[str] = "r"
    ) -> np.ndarray:
        """Loads dense representation embeddings matrix (optionally zero-copy memmapped)."""
        emb_path = self.task_dir(domain, task_id) / "embeddings" / f"{encoder_id}.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings '{encoder_id}' not found at {emb_path}")
        return np.load(emb_path, mmap_mode=mmap_mode)

    def list_tasks(self, domain: Optional[str] = None) -> List[tuple[str, str]]:
        """Lists available (domain, task_id) pairs."""
        tasks = []
        domains = [domain] if domain else [d.name for d in self.tasks_root.iterdir() if d.is_dir()]
        for d in sorted(domains):
            ddir = self.tasks_root / d
            if not ddir.is_dir():
                continue
            for tdir in sorted(ddir.iterdir()):
                if tdir.is_dir() and (tdir / "metadata.json").exists():
                    tasks.append((d, tdir.name))
        return tasks
