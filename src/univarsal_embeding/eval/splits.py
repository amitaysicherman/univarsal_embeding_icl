"""Universal Generalization Splits (Seen, Unseen Tasks, Unseen Encoders, Unseen Both, Unseen Domain)."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class GeneralizationPartitions:
    train_tasks: List[str]
    test_tasks: List[str]
    train_encoders: List[str]
    test_encoders: List[str]
    domain: str


def deterministic_task_split(task_ids: List[str], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministically partition tasks into train (80%) and held-out test (20%)."""
    sorted_tasks = sorted(task_ids)
    hashed = sorted(
        sorted_tasks,
        key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest(),
    )
    n_test = max(1, int(len(sorted_tasks) * test_ratio))
    test_tasks = sorted(hashed[:n_test])
    train_tasks = sorted(hashed[n_test:])
    return train_tasks, test_tasks


def get_kfold_task_split(task_ids: List[str], fold: int = 0, num_folds: int = 5, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministically partition tasks into num_folds non-overlapping folds.
    For fold k, returns (train_tasks, test_tasks) where test_tasks is the k-th fold.
    Across all folds, 100% of tasks are evaluated as test_tasks exactly once.
    """
    sorted_tasks = sorted(task_ids)
    hashed = sorted(
        sorted_tasks,
        key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest(),
    )
    buckets: List[List[str]] = [[] for _ in range(num_folds)]
    for i, t in enumerate(hashed):
        buckets[i % num_folds].append(t)

    test_tasks = sorted(buckets[fold % num_folds])
    train_tasks = sorted([t for j, b in enumerate(buckets) if j != (fold % num_folds) for t in b])
    return train_tasks, test_tasks


def get_encoder_split(domain_encoders: List[str]) -> Tuple[List[str], List[str]]:
    """Partition domain encoders into train encoders (base 3-4) and held-out test encoders (1-2 SOTA)."""
    if len(domain_encoders) >= 5:
        train_encoders = domain_encoders[:-1]
        test_encoders = domain_encoders[-1:]
    elif len(domain_encoders) == 4:
        train_encoders = domain_encoders[:3]
        test_encoders = domain_encoders[3:]
    else:
        train_encoders = domain_encoders[:2]
        test_encoders = domain_encoders[2:]
    return train_encoders, test_encoders


def get_kfold_encoder_split(domain_encoders: List[str], fold: int = 0, num_folds: int = 5) -> Tuple[List[str], List[str]]:
    """Rotate the held-out test encoder across folds.
    Across all num_folds folds, 100% of encoders are evaluated as test_encoders.
    """
    buckets: List[List[str]] = [[] for _ in range(num_folds)]
    for i, e in enumerate(domain_encoders):
        buckets[i % num_folds].append(e)

    test_encoders = sorted(buckets[fold % num_folds])
    train_encoders = sorted([e for j, b in enumerate(buckets) if j != (fold % num_folds) for e in b])
    return train_encoders, test_encoders


class UniversalBenchmarkSplits:
    """Benchmark manager defining the 5 canonical generalization evaluation regimes,
    supporting both single-split (80/20) and full 5-Fold Cross-Generalization.
    """

    def __init__(
        self,
        domain_to_tasks: Dict[str, List[str]],
        domain_to_encoders: Dict[str, List[str]],
        test_task_ratio: float = 0.2,
        seed: int = 42,
        fold: Optional[int] = None,
        num_folds: int = 5,
    ) -> None:
        self.domain_to_tasks = domain_to_tasks
        self.domain_to_encoders = domain_to_encoders
        self.domains = sorted(domain_to_tasks.keys())
        self.test_task_ratio = test_task_ratio
        self.seed = seed
        self.fold = fold
        self.num_folds = num_folds

        self.partitions: Dict[str, GeneralizationPartitions] = {}
        for d in self.domains:
            if fold is not None:
                tr_tasks, te_tasks = get_kfold_task_split(
                    domain_to_tasks[d], fold=fold, num_folds=num_folds, seed=seed
                )
                tr_encs, te_encs = get_kfold_encoder_split(
                    domain_to_encoders[d], fold=fold, num_folds=num_folds
                )
            else:
                tr_tasks, te_tasks = deterministic_task_split(
                    domain_to_tasks[d], test_ratio=test_task_ratio, seed=seed
                )
                tr_encs, te_encs = get_encoder_split(domain_to_encoders[d])

            self.partitions[d] = GeneralizationPartitions(
                train_tasks=tr_tasks,
                test_tasks=te_tasks,
                train_encoders=tr_encs,
                test_encoders=te_encs,
                domain=d,
            )

    def get_eval_pairs(
        self, regime: str, holdout_domain: str | List[str] | None = None
    ) -> List[Tuple[str, str, str]]:
        """Returns list of (domain, task_id, encoder_id) for the specified evaluation regime.

        Regimes:
        - 'seen': Tasks in train_tasks, Encoders in train_encoders
        - 'unseen_tasks': Tasks in test_tasks, Encoders in train_encoders
        - 'unseen_encoders': Tasks in train_tasks, Encoders in test_encoders
        - 'unseen_both': Tasks in test_tasks, Encoders in test_encoders
        - 'unseen_domain': All tasks & encoders of the held-out domain(s) (trained on the
          remaining domains). holdout_domain may be a single domain name or a list of
          domain names (for grouped leave-N-domains-out runs).
        """
        pairs = []
        regime = regime.lower()

        if regime == "unseen_domain":
            if holdout_domain is None:
                raise ValueError("Valid holdout_domain required for unseen_domain regime")
            holdout_domains = [holdout_domain] if isinstance(holdout_domain, str) else list(holdout_domain)
            for hd in holdout_domains:
                if hd not in self.domain_to_tasks:
                    raise ValueError(f"Invalid holdout_domain: {hd}")
                for task_id in self.domain_to_tasks[hd]:
                    for enc_id in self.domain_to_encoders.get(hd, []):
                        pairs.append((hd, task_id, enc_id))
            return pairs

        for d, p in self.partitions.items():
            if regime == "seen":
                tasks, encs = p.train_tasks, p.train_encoders
            elif regime == "unseen_tasks":
                tasks, encs = p.test_tasks, p.train_encoders
            elif regime == "unseen_encoders":
                tasks, encs = p.train_tasks, p.test_encoders
            elif regime == "unseen_both":
                tasks, encs = p.test_tasks, p.test_encoders
            else:
                raise ValueError(f"Unknown generalization regime: {regime}")

            for t in tasks:
                for e in encs:
                    pairs.append((d, t, e))

        return pairs
