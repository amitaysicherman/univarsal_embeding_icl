"""Central registry and dispatch for all multi-domain task fetchers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.schema import Modality, TaskData
from .audio import fetch_all_audio_tasks
from .graphs import fetch_all_graph_tasks
from .molecules import fetch_all_molecule_tasks
from .proteins import fetch_all_protein_tasks
from .text import fetch_all_text_tasks
from .timeseries import fetch_all_timeseries_tasks
from .vision import fetch_all_vision_tasks

logger = logging.getLogger(__name__)


def fetch_tasks(
    domain: str = "all",
    cache_root: Optional[Path | str] = None,
    smoke: bool = False,
    limit_per_domain: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> Dict[str, List[TaskData]]:
    """Fetches tasks for one or all modalities with canonical train/val/test splits."""
    cache_dir = Path(cache_root) if cache_root else Path("data/raw")
    img_dir = cache_dir / "images"
    audio_dir = cache_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    fetchers = {
        Modality.MOLECULES.value: lambda: fetch_all_molecule_tasks(smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.TEXT.value: lambda: fetch_all_text_tasks(smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.PROTEINS.value: lambda: fetch_all_protein_tasks(smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.VISION.value: lambda: fetch_all_vision_tasks(image_cache_dir=img_dir, smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.AUDIO.value: lambda: fetch_all_audio_tasks(audio_cache_dir=audio_dir, smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.GRAPHS.value: lambda: fetch_all_graph_tasks(smoke=smoke, limit=limit_per_domain, store=store, force=force),
        Modality.TIMESERIES.value: lambda: fetch_all_timeseries_tasks(smoke=smoke, limit=limit_per_domain, store=store, force=force),
    }

    target_domains = list(fetchers.keys()) if domain == "all" else [domain]
    results: Dict[str, List[TaskData]] = {}

    for d in target_domains:
        if d not in fetchers:
            raise ValueError(f"Unknown domain: '{d}'. Available: {list(fetchers.keys())}")
        logger.info(f"===> Fetching domain '{d}' (smoke={smoke})...")
        tasks = fetchers[d]()
        results[d] = tasks
        logger.info(f"Loaded {len(tasks)} tasks for domain '{d}'")

    return results
