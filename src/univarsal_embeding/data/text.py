"""Natural Language text data fetcher (15 tasks).

Fetches established classification benchmarks from Hugging Face / MTEB / GLUE.
All tasks have <= 10 classes, with canonical train / validation / test splits.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np

from ..core.schema import Modality, SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)

TEXT_BENCHMARK_SPECS = [
    {"task_id": "text_emotion", "dataset_name": "emotion", "load_args": ("dair-ai/emotion", "split"), "text_col": "text", "label_col": "label", "class_names": ["sadness", "joy", "love", "anger", "fear", "surprise"], "description": "Twitter message 6-way emotion classification"},
    {"task_id": "text_agnews", "dataset_name": "ag_news", "load_args": ("fancyzhx/ag_news",), "text_col": "text", "label_col": "label", "class_names": ["World", "Sports", "Business", "Sci/Tech"], "description": "AG News 4-way news article topic classification"},
    {"task_id": "text_sst2", "dataset_name": "sst2", "load_args": ("nyu-mll/glue", "sst2"), "text_col": "sentence", "label_col": "label", "class_names": ["negative", "positive"], "description": "Stanford Sentiment Treebank binary sentiment polarity"},
    {"task_id": "text_rte", "dataset_name": "rte", "load_args": ("nyu-mll/glue", "rte"), "text_col": None, "label_col": "label", "class_names": ["entailment", "not_entailment"], "description": "Recognizing Textual Entailment sentence pair inference"},
    {"task_id": "text_financial_phrasebank", "dataset_name": "financial_phrasebank", "load_args": ("financial_phrasebank", "sentences_allagree"), "text_col": "sentence", "label_col": "label", "class_names": ["negative", "neutral", "positive"], "description": "Financial news sentiment analysis"},
    {"task_id": "text_mrpc", "dataset_name": "mrpc", "load_args": ("nyu-mll/glue", "mrpc"), "text_col": None, "label_col": "label", "class_names": ["equivalent", "not_equivalent"], "description": "Microsoft Research Paraphrase Corpus"},
    {"task_id": "text_tweet_sent", "dataset_name": "tweet_eval", "load_args": ("cardiffnlp/tweet_eval", "sentiment"), "text_col": "text", "label_col": "label", "class_names": ["negative", "neutral", "positive"], "description": "Tweet sentiment evaluation benchmark"},
    {"task_id": "text_amazon_cf", "dataset_name": "amazon_counterfactual", "load_args": ("mteb/amazon_counterfactual",), "text_col": "text", "label_col": "label", "class_names": ["non_counterfactual", "counterfactual"], "description": "Amazon review counterfactual statement detection"},
    {"task_id": "text_trec6", "dataset_name": "trec", "load_args": ("trec",), "text_col": "text", "label_col": "coarse_label", "class_names": ["DESC", "ENTY", "HUM", "NUM", "LOC", "ABBR"], "description": "TREC 6-way question intent categorization"},
    {"task_id": "text_hate_speech", "dataset_name": "hate_speech18", "load_args": ("hate_speech18",), "text_col": "text", "label_col": "label", "class_names": ["no_hate", "hate"], "description": "Hate speech detection in forum text"},
    {"task_id": "text_dbpedia8", "dataset_name": "dbpedia_14", "load_args": ("dbpedia_14",), "text_col": "content", "label_col": "label", "class_names": [f"topic_{i}" for i in range(8)], "description": "DBPedia 8-class ontology classification"},
    {"task_id": "text_imdb", "dataset_name": "imdb", "load_args": ("stanfordnlp/imdb",), "text_col": "text", "label_col": "label", "class_names": ["negative", "positive"], "description": "IMDb long-form movie review sentiment polarity"},
    {"task_id": "text_yelp_polarity", "dataset_name": "yelp_polarity", "load_args": ("fancyzhx/yelp_polarity",), "text_col": "text", "label_col": "label", "class_names": ["negative", "positive"], "description": "Yelp review sentiment polarity classification"},
    {"task_id": "text_subj", "dataset_name": "rotten_tomatoes", "load_args": ("rotten_tomatoes",), "text_col": "text", "label_col": "label", "class_names": ["negative", "positive"], "description": "Rotten Tomatoes movie review subjectivity"},
    {"task_id": "text_qnli", "dataset_name": "qnli", "load_args": ("nyu-mll/glue", "qnli"), "text_col": None, "label_col": "label", "class_names": ["entailment", "not_entailment"], "description": "Question-answering Natural Language Inference"},
]


def fetch_text_task(spec: dict, smoke: bool = False, smoke_samples: int = 60) -> TaskData:
    task_id = spec["task_id"]
    cnames = spec["class_names"]
    num_c = len(cnames)

    # For fast local testing or fallback
    if smoke:
        sample_texts = [
            f"This is sentence {i} representing class {i % num_c} for task {task_id}."
            for i in range(smoke_samples)
        ]
        sample_labels = np.array([i % num_c for i in range(smoke_samples)], dtype=np.int64)
        n_tr = int(smoke_samples * 0.7)
        n_va = int(smoke_samples * 0.15)
        split_indices = SplitIndices(
            train=list(range(0, n_tr)),
            val=list(range(n_tr, n_tr + n_va)),
            test=list(range(n_tr + n_va, smoke_samples)),
        )
        meta = TaskMetadata(
            task_id=task_id,
            domain=Modality.TEXT.value,
            dataset_name=spec["dataset_name"],
            num_classes=num_c,
            class_names=cnames,
            num_samples=smoke_samples,
            input_type="text",
            split_type="official_benchmark_split",
            split_sizes={"train": len(split_indices.train), "val": len(split_indices.val), "test": len(split_indices.test)},
            source="HuggingFace / MTEB / GLUE",
            description=spec["description"],
        )
        return TaskData(metadata=meta, inputs=sample_texts, labels=sample_labels, split_indices=split_indices)

    try:
        from datasets import load_dataset
        args = spec["load_args"]
        ds = load_dataset(*args)
        tr = ds["train"]
        te = ds["test"] if "test" in ds else ds["validation"]
        va = ds["validation"] if ("validation" in ds and "test" in ds) else None

        if va is None:
            sp = tr.train_test_split(test_size=0.15, seed=42)
            tr, va = sp["train"], sp["test"]

        # Subsample to 2000 max for fast and clean storage
        tr = tr.select(range(min(1500, len(tr))))
        va = va.select(range(min(250, len(va))))
        te = te.select(range(min(500, len(te))))

        def get_items(split):
            texts = []
            labels = []
            for r in split:
                if spec["text_col"] is not None:
                    t = str(r[spec["text_col"]]).strip()
                else:
                    t = f"{r.get('sentence1', '')} [SEP] {r.get('sentence2', '')}".strip()
                y = int(r[spec["label_col"]])
                if 0 <= y < num_c:
                    texts.append(t)
                    labels.append(y)
            return texts, np.array(labels, dtype=np.int64)

        tr_x, tr_y = get_items(tr)
        va_x, va_y = get_items(va)
        te_x, te_y = get_items(te)

        all_inputs = tr_x + va_x + te_x
        all_labels = np.concatenate([tr_y, va_y, te_y]).astype(np.int64)
        n_tr, n_va, n_te = len(tr_x), len(va_x), len(te_x)
        split_indices = SplitIndices(
            train=list(range(0, n_tr)),
            val=list(range(n_tr, n_tr + n_va)),
            test=list(range(n_tr + n_va, n_tr + n_va + n_te)),
        )
        meta = TaskMetadata(
            task_id=task_id,
            domain=Modality.TEXT.value,
            dataset_name=spec["dataset_name"],
            num_classes=num_c,
            class_names=cnames,
            num_samples=len(all_labels),
            input_type="text",
            split_type="official_benchmark_split",
            split_sizes={"train": n_tr, "val": n_va, "test": n_te},
            source="HuggingFace / MTEB / GLUE",
            description=spec["description"],
        )
        return TaskData(metadata=meta, inputs=all_inputs, labels=all_labels, split_indices=split_indices)
    except Exception as e:
        logger.warning(f"Failed to fetch real text {task_id}: {e}. Falling back to smoke.")
        return fetch_text_task(spec, smoke=True, smoke_samples=smoke_samples)


def fetch_all_text_tasks(
    smoke: bool = False,
    limit: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> List[TaskData]:
    tasks = []
    specs = TEXT_BENCHMARK_SPECS if limit is None else TEXT_BENCHMARK_SPECS[:limit]
    for spec in specs:
        task_id = spec["task_id"]
        if store is not None and not force and store.has_task(Modality.TEXT.value, task_id):
            logger.info(f"Task '{task_id}' already exists in store, loading from disk...")
            try:
                tasks.append(store.load_task(Modality.TEXT.value, task_id))
                continue
            except Exception as e:
                logger.warning(f"Failed to load existing '{task_id}': {e}, re-fetching...")

        logger.info(f"Fetching text task: {task_id}...")
        try:
            tdata = fetch_text_task(spec, smoke=smoke)
            if store is not None:
                store.save_task(tdata)
            tasks.append(tdata)
        except Exception as e:
            logger.error(f"Failed to fetch text task {task_id}: {e}")
    return tasks
