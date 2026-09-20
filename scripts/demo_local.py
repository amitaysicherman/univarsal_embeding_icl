#!/usr/bin/env python3
"""End-to-End standalone demo of data fetching, split verification, and embedding extraction."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
import numpy as np

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from univarsal_embeding.core.store import TaskStore
from univarsal_embeding.data.registry import fetch_tasks
from univarsal_embeding.encoders.registry import DOMAIN_TO_ENCODERS, get_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def run_demo():
    print("=" * 80)
    print("UNIVERSAL MULTIMODAL EMBEDDING BENCHMARK - STANDALONE LOCAL DEMO")
    print("=" * 80)

    demo_root = Path("data/demo")
    store = TaskStore(data_root=demo_root)

    # 1. Fetch smoke data across all domains
    logger.info("Step 1: Fetching smoke datasets across all 7 modalities...")
    all_domain_tasks = fetch_tasks(
        domain="all",
        cache_root=demo_root / "raw",
        smoke=True,
        limit_per_domain=2,
    )

    # 2. Verify split integrity and save to store
    logger.info("Step 2: Verifying split integrity and storing canonical formats...")
    saved_tasks = []
    for domain, tasks in all_domain_tasks.items():
        for tdata in tasks:
            # Check non-overlap of train, val, test
            s = tdata.split_indices
            tr_set, va_set, te_set = set(s.train), set(s.val), set(s.test)
            assert tr_set.isdisjoint(va_set), f"Overlap between train and val in {tdata.metadata.task_id}"
            assert tr_set.isdisjoint(te_set), f"Overlap between train and test in {tdata.metadata.task_id}"
            assert va_set.isdisjoint(te_set), f"Overlap between val and test in {tdata.metadata.task_id}"
            assert len(tr_set) + len(va_set) + len(te_set) == len(tdata.labels), "Split sum mismatch"
            assert tdata.metadata.num_classes <= 10, f"Classes > 10 in {tdata.metadata.task_id}"

            tdir = store.save_task(tdata)
            saved_tasks.append((domain, tdata.metadata.task_id))

    print(f"\nSuccessfully verified and saved {len(saved_tasks)} tasks across 7 modalities!")

    # 3. Embedding extraction demonstration
    logger.info("Step 3: Extracting embeddings via encoders...")
    for domain, task_id in saved_tasks:
        task_data = store.load_task(domain, task_id)
        encoders = DOMAIN_TO_ENCODERS.get(domain, ["pseudo_encoder"])
        encoder_id = encoders[0]

        # Use dummy mode for instant local testing without downloading multi-gigabyte models
        encoder = get_encoder(encoder_id, domain=domain, use_dummy=True, dummy_dim=512)
        embs = encoder.encode(task_data.inputs)
        assert embs.shape == (len(task_data.labels), 512)

        store.save_embeddings(domain, task_id, encoder_id, embs)

    # 4. In-Context random projection classification demo
    logger.info("Step 4: Zero-gradient dynamic random projection ICL test...")
    proj_dim = 256
    for domain, task_id in saved_tasks[:3]:
        task_data = store.load_task(domain, task_id)
        encoder_id = DOMAIN_TO_ENCODERS.get(domain, ["pseudo_encoder"])[0]
        embs = store.load_embeddings(domain, task_id, encoder_id)

        # Retrieve explicit train (support) and test (query) splits
        tr_idx = task_data.split_indices.train
        te_idx = task_data.split_indices.test

        support_x, support_y = embs[tr_idx], task_data.labels[tr_idx]
        query_x, query_y = embs[te_idx], task_data.labels[te_idx]

        # Dynamic Johnson-Lindenstrauss random projection R ~ N(0, 1/d_proj)
        d_in = embs.shape[1]
        R = np.random.randn(d_in, proj_dim).astype(np.float32) / math.sqrt(proj_dim)

        proj_support_x = support_x @ R
        proj_query_x = query_x @ R

        # Cosine / nearest centroid similarity in projected space
        classes = np.unique(support_y)
        centroids = np.array([proj_support_x[support_y == c].mean(axis=0) for c in classes])
        # Compute distances to centroids
        dists = np.linalg.norm(proj_query_x[:, None, :] - centroids[None, :, :], axis=2)
        preds = classes[np.argmin(dists, axis=1)]
        acc = (preds == query_y).mean()

        print(
            f"  - [{domain.upper()}] Task '{task_id}': Support N={len(tr_idx)}, "
            f"Query N={len(te_idx)}, Proj dim={proj_dim} -> Accuracy = {acc * 100:.1f}%"
        )

    print("\n" + "=" * 80)
    print("ALL DEMO VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print(f"Data files stored standalone in: {demo_root.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
