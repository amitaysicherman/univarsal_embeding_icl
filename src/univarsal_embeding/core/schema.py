"""Core schemas and data models for standalone universal embedding benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class Modality(str, Enum):
    MOLECULES = "molecules"
    PROTEINS = "proteins"
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    GRAPHS = "graphs"
    TIMESERIES = "timeseries"


@dataclass
class SplitIndices:
    """Explicit indices partitioning the samples into train, validation, and test sets."""
    train: List[int]
    val: List[int]
    test: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, data: Dict[str, List[int]]) -> "SplitIndices":
        return cls(
            train=list(data.get("train", [])),
            val=list(data.get("val", [])),
            test=list(data.get("test", [])),
        )

    def to_folds_array(self, total_samples: int) -> np.ndarray:
        """Returns integer fold array: 0=test, 1=val, 2=train (compatible with TabICL)."""
        folds = np.zeros(total_samples, dtype=np.int8)
        folds[self.test] = 0
        folds[self.val] = 1
        folds[self.train] = 2
        return folds


@dataclass
class TaskMetadata:
    """Rich metadata detailing task provenance, class space, and split configuration."""
    task_id: str
    domain: str
    dataset_name: str
    num_classes: int
    class_names: List[str]
    num_samples: int
    input_type: str  # "smiles", "sequence", "image_path", "text", "audio_path", "graph", "timeseries"
    split_type: str  # e.g., "bemis_murcko_scaffold", "sequence_identity_30", "official_split", "speaker_hash"
    split_sizes: Dict[str, int]
    source: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskMetadata":
        import inspect
        valid_fields = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "dataset_name" not in filtered:
            filtered["dataset_name"] = filtered.get("task_id", "")
        if "input_type" not in filtered:
            filtered["input_type"] = filtered.get("domain", "")
        if "split_sizes" not in filtered:
            filtered["split_sizes"] = d.get("folds_distribution", {})
        if "split_type" not in filtered:
            filtered["split_type"] = d.get("task_type", "unspecified")
        return cls(**filtered)


@dataclass
class TaskData:
    """Container holding inputs, ground-truth labels, and explicit split indices."""
    metadata: TaskMetadata
    inputs: List[Any]
    labels: np.ndarray
    split_indices: SplitIndices

    def __post_init__(self):
        if len(self.inputs) != len(self.labels):
            raise ValueError(f"Mismatch: {len(self.inputs)} inputs vs {len(self.labels)} labels")
        total_split = len(self.split_indices.train) + len(self.split_indices.val) + len(self.split_indices.test)
        if total_split != len(self.labels):
            raise ValueError(f"Split count {total_split} does not equal total samples {len(self.labels)}")
