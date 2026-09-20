#!/usr/bin/env python3
"""Batch extraction of representation embeddings across all tasks and 4 SOTA encoders per domain."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
import torch

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from univarsal_embeding.core.store import TaskStore
from univarsal_embeding.encoders.registry import DOMAIN_TO_ENCODERS, get_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract representation embeddings across all tasks.")
    parser.add_argument("--domain", type=str, default="all", help="Target domain (default: all)")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy pseudo-encoders for dry-run testing")

    args = parser.parse_args()
    store = TaskStore(data_root=args.data_root)

    tasks = store.list_tasks(domain=None if args.domain == "all" else args.domain)
    logger.info(f"Discovered {len(tasks)} tasks to encode (domain={args.domain}, device={args.device})")

    total_start = time.time()
    extracted_count = 0
    skipped_count = 0

    for domain, task_id in tasks:
        task_data = store.load_task(domain, task_id)
        encoders = DOMAIN_TO_ENCODERS.get(domain, [])
        logger.info(f"\n---> Task [{domain.upper()}] '{task_id}' (N={len(task_data.inputs)}, C={task_data.metadata.num_classes})")

        for encoder_id in encoders:
            emb_path = store.task_dir(domain, task_id) / "embeddings" / f"{encoder_id}.npy"
            if emb_path.exists() and not args.overwrite:
                logger.info(f"  [SKIPPED] Encoder '{encoder_id}' already exists at {emb_path}")
                skipped_count += 1
                continue

            logger.info(f"  [ENCODING] Initializing '{encoder_id}' on {args.device}...")
            t0 = time.time()
            try:
                encoder = get_encoder(encoder_id, domain=domain, device=args.device, use_dummy=args.use_dummy)
                embs = encoder.encode(task_data.inputs, batch_size=args.batch_size)
                elapsed = time.time() - t0

                # CRITICAL: Validate embeddings are real (not dummy/degenerate)
                import numpy as np
                if not isinstance(embs, np.ndarray):
                    raise TypeError(f"Encoder returned {type(embs)}, expected numpy array")

                if embs.shape[0] != len(task_data.inputs):
                    raise ValueError(
                        f"Embedding count mismatch: got {embs.shape[0]}, expected {len(task_data.inputs)}"
                    )

                if np.any(np.isnan(embs)) or np.any(np.isinf(embs)):
                    raise ValueError("Embeddings contain NaN or Inf values (degenerate/failed extraction)")

                # Check variance (degenerate if ~0)
                var = np.var(embs)
                if var < 1e-7:
                    raise ValueError(
                        f"Embeddings have degenerate variance {var:.2e} (likely dummy or all-zeros)"
                    )

                # Warn if unusual norm distribution
                norms = np.linalg.norm(embs, axis=1)
                mean_norm = np.mean(norms)
                if mean_norm < 0.01 or mean_norm > 1000.0:
                    logger.warning(
                        f"Embedding norm distribution unusual: mean={mean_norm:.4f} "
                        f"(expected ~1-10 range for foundation models)"
                    )

                store.save_embeddings(domain, task_id, encoder_id, embs)
                logger.info(f"  [DONE] '{encoder_id}' shape={embs.shape} mean_norm={mean_norm:.4f} variance={var:.2e} in {elapsed:.2f}s")
                extracted_count += 1
            except Exception as e:
                logger.exception(f"  [FAILED] Encoder '{encoder_id}' on task '{task_id}': {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("EMBEDDING EXTRACTION SUMMARY")
    print(f"Total time elapsed: {total_time / 60:.2f} minutes")
    print(f"Total embeddings extracted: {extracted_count}")
    print(f"Total skipped (already existed): {skipped_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
