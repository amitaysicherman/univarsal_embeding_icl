"""Unit and smoke tests for the standalone universal embedding package."""

import numpy as np
import pytest
from univarsal_embeding.core.schema import SplitIndices, TaskData, TaskMetadata, Modality
from univarsal_embeding.core.store import TaskStore
from univarsal_embeding.encoders.base import DummyEncoder
from univarsal_embeding.data.registry import fetch_tasks


def test_schema_and_splits():
    split = SplitIndices(train=[0, 1], val=[2], test=[3, 4])
    folds = split.to_folds_array(5)
    assert (folds == np.array([2, 2, 1, 0, 0])).all()


def test_task_store(tmp_path):
    store = TaskStore(data_root=tmp_path)
    split = SplitIndices(train=[0, 1], val=[2], test=[3])
    meta = TaskMetadata(
        task_id="test_task",
        domain="molecules",
        dataset_name="test_chem",
        num_classes=2,
        class_names=["inactive", "active"],
        num_samples=4,
        input_type="smiles",
        split_type="scaffold",
        split_sizes={"train": 2, "val": 1, "test": 1},
        source="unit_test",
        description="test task",
    )
    task_data = TaskData(
        metadata=meta,
        inputs=["C", "CC", "CCC", "CCCC"],
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        split_indices=split,
    )
    store.save_task(task_data)

    loaded = store.load_task("molecules", "test_task")
    assert loaded.metadata.task_id == "test_task"
    assert len(loaded.inputs) == 4
    assert (loaded.labels == np.array([0, 0, 1, 1])).all()
    assert loaded.split_indices.train == [0, 1]

    # Test embeddings
    embs = np.random.randn(4, 128).astype(np.float32)
    store.save_embeddings("molecules", "test_task", "dummy_enc", embs)
    loaded_embs = store.load_embeddings("molecules", "test_task", "dummy_enc")
    assert loaded_embs.shape == (4, 128)


def test_dummy_encoder():
    enc = DummyEncoder(encoder_id="dummy", domain="text", embedding_dim=256)
    out = enc.encode(["hello world", "test sentence"])
    assert out.shape == (2, 256)
    assert out.dtype == np.float32
