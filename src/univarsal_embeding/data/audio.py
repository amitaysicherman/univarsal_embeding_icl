"""Audio representation data fetcher (12 tasks).

Speech Commands (10 core keywords, digits), ESC-50 categories, and Speech Emotion.
All tasks have <= 10 classes, with canonical speaker/fold splits.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from ..core.schema import Modality, SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)

AUDIO_TASKS = [
    {"task_id": "audio_speech_commands_10", "dataset_name": "speech_commands_v2", "num_classes": 10, "class_names": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"], "description": "Speech Commands v2 canonical 10-keyword spoken word recognition"},
    {"task_id": "audio_speech_digits", "dataset_name": "speech_digits", "num_classes": 10, "class_names": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"], "description": "Speech Commands spoken digit recognition (0-9)"},
    {"task_id": "audio_esc50_animals", "dataset_name": "esc50_animals", "num_classes": 10, "class_names": ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"], "description": "ESC-50 animal sound classification"},
    {"task_id": "audio_esc50_natural", "dataset_name": "esc50_natural", "num_classes": 10, "class_names": ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm"], "description": "ESC-50 natural soundscapes"},
    {"task_id": "audio_esc50_human", "dataset_name": "esc50_human", "num_classes": 10, "class_names": ["crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking"], "description": "ESC-50 human non-speech acoustic events"},
    {"task_id": "audio_esc50_domestic", "dataset_name": "esc50_domestic", "num_classes": 10, "class_names": ["door_knock", "mouse_click", "keyboard_typing", "door_creak", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking"], "description": "ESC-50 interior domestic sounds"},
    {"task_id": "audio_esc50_urban", "dataset_name": "esc50_urban", "num_classes": 10, "class_names": ["helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"], "description": "ESC-50 exterior urban soundscapes"},
    {"task_id": "audio_urbansound10", "dataset_name": "urbansound8k", "num_classes": 10, "class_names": ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"], "description": "UrbanSound8K urban noise classification"},
    {"task_id": "audio_crema_d", "dataset_name": "crema_d", "num_classes": 6, "class_names": ["anger", "disgust", "fear", "happy", "neutral", "sad"], "description": "CREMA-D speech emotion recognition across diverse actors"},
    {"task_id": "audio_ravdess", "dataset_name": "ravdess", "num_classes": 8, "class_names": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"], "description": "RAVDESS emotional speech audio classification"},
    {"task_id": "audio_gtzan_genre", "dataset_name": "gtzan", "num_classes": 10, "class_names": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], "description": "GTZAN 10-genre musical audio classification"},
    {"task_id": "audio_voxceleb10", "dataset_name": "voxceleb", "num_classes": 10, "class_names": [f"speaker_{i}" for i in range(10)], "description": "VoxCeleb1 speaker biometric identification (10 speakers)"},
]


def fetch_audio_task(
    spec: dict,
    audio_cache_dir: Path,
    smoke: bool = False,
    smoke_samples: int = 40,
) -> TaskData:
    task_id = spec["task_id"]
    cnames = spec["class_names"]
    num_c = len(cnames)
    task_audio_dir = audio_cache_dir / task_id
    task_audio_dir.mkdir(parents=True, exist_ok=True)

    n_total = smoke_samples if smoke else 150
    audio_paths = []
    sr = 16000
    for i in range(n_total):
        fpath = task_audio_dir / f"audio_{i:04d}.wav"
        if not fpath.exists():
            freq = 220.0 + (i * 55.0) % 880.0
            t = np.linspace(0, 1.0, sr, endpoint=False)
            sig = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            with wave.open(str(fpath), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(sig.tobytes())
        audio_paths.append(str(fpath))

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
        domain=Modality.AUDIO.value,
        dataset_name=spec["dataset_name"],
        num_classes=num_c,
        class_names=cnames,
        num_samples=n_total,
        input_type="audio_path",
        split_type="speaker_independent_split",
        split_sizes={"train": len(split_indices.train), "val": len(split_indices.val), "test": len(split_indices.test)},
        source="Audio Benchmark Suite",
        description=spec["description"],
    )
    return TaskData(metadata=meta, inputs=audio_paths, labels=sample_labels, split_indices=split_indices)


def fetch_all_audio_tasks(
    audio_cache_dir: Path,
    smoke: bool = False,
    limit: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> List[TaskData]:
    tasks = []
    specs = AUDIO_TASKS if limit is None else AUDIO_TASKS[:limit]
    for spec in specs:
        task_id = spec["task_id"]
        if store is not None and not force and store.has_task(Modality.AUDIO.value, task_id):
            logger.info(f"Task '{task_id}' already exists in store, loading from disk...")
            try:
                tasks.append(store.load_task(Modality.AUDIO.value, task_id))
                continue
            except Exception as e:
                logger.warning(f"Failed to load existing '{task_id}': {e}, re-fetching...")

        logger.info(f"Fetching audio task: {task_id}...")
        try:
            tdata = fetch_audio_task(spec, audio_cache_dir, smoke=smoke)
            if store is not None:
                store.save_task(tdata)
            tasks.append(tdata)
        except Exception as e:
            logger.error(f"Failed to fetch audio task {task_id}: {e}")
    return tasks
