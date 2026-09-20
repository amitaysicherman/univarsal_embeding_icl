#!/usr/bin/env python3
"""CLI script to fetch and prepare all cross-domain datasets with canonical splits."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from univarsal_embeding.core.store import TaskStore
from univarsal_embeding.data.registry import fetch_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("univarsal_embeding.fetch_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch cross-domain benchmark datasets with canonical splits.")
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "molecules", "proteins", "vision", "text", "audio", "graphs", "timeseries"],
        help="Domain to fetch (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Data storage root directory (default: data)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke fetch with reduced samples for quick local verification",
    )
    parser.add_argument(
        "--limit-per-domain",
        type=int,
        default=None,
        help="Limit number of tasks per domain (default: all)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetching and overwriting existing tasks",
    )

    args = parser.parse_args()
    store = TaskStore(data_root=args.output_dir)

    logger.info(f"Starting data acquisition (domain={args.domain}, smoke={args.smoke}, force={args.force})...")
    domain_tasks = fetch_tasks(
        domain=args.domain,
        cache_root=args.output_dir / "raw",
        smoke=args.smoke,
        limit_per_domain=args.limit_per_domain,
        store=store,
        force=args.force,
    )

    # Collect summary from store
    target_domains = list(domain_tasks.keys())
    summary = []
    for d in target_domains:
        task_pairs = store.list_tasks(domain=d)
        for _, tid in task_pairs:
            try:
                tdata = store.load_task(d, tid)
                meta = tdata.metadata
                summary.append({
                    "domain": d,
                    "task_id": meta.task_id,
                    "samples": meta.num_samples,
                    "classes": meta.num_classes,
                    "train": meta.split_sizes.get("train", 0),
                    "val": meta.split_sizes.get("val", 0),
                    "test": meta.split_sizes.get("test", 0),
                    "split_type": meta.split_type,
                })
            except Exception as e:
                logger.warning(f"Error loading summary for {d}/{tid}: {e}")

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'DOMAIN':<12} | {'TASK ID':<28} | {'N':<6} | {'C':<3} | {'TRAIN':<6} | {'VAL':<5} | {'TEST':<5} | {'SPLIT TYPE'}")
    print("-" * 105)
    for s in summary:
        print(
            f"{s['domain']:<12} | {s['task_id']:<28} | {s['samples']:<6} | {s['classes']:<3} | "
            f"{s['train']:<6} | {s['val']:<5} | {s['test']:<5} | {s['split_type']}"
        )
    print("=" * 105)
    print(f"Total tasks available in store: {len(summary)} under {store.tasks_root}\n")


if __name__ == "__main__":
    main()
