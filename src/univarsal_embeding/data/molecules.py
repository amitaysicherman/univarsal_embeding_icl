"""Molecules data fetcher with canonical Bemis-Murcko scaffold splits (24 tasks).

Uses Therapeutics Data Commons (TDC) official scaffold splits.
All tasks are binary (C=2), with zero scaffold leakage.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np

from ..core.schema import Modality, SplitIndices, TaskData, TaskMetadata

logger = logging.getLogger(__name__)

# Curated 24 TDC tasks with verified native dataset names and canonical Bemis-Murcko scaffold splits
TDC_TOX_TASKS = [
    ("ames", "Ames bacterial reverse mutagenicity", ["non_mutagenic", "mutagenic"]),
    ("herg", "hERG potassium cardiac channel block", ["inactive", "blocker"]),
    ("dili", "Drug-induced liver injury toxicity", ["safe", "toxic"]),
    ("skin_reaction", "Skin sensitization contact hazard", ["safe", "sensitizer"]),
    ("carcinogens_lagunin", "Rodent carcinogenicity hazard", ["non_carcinogenic", "carcinogenic"]),
    ("clintox", "FDA approval vs clinical trial toxicity failure", ["failed_tox", "fda_approved"]),
]

TDC_ADME_TASKS = [
    ("cyp2d6_veith", "CYP2D6 enzyme inhibition (Veith)", ["non_inhibitor", "inhibitor"]),
    ("cyp3a4_veith", "CYP3A4 enzyme inhibition (Veith)", ["non_inhibitor", "inhibitor"]),
    ("cyp2c9_veith", "CYP2C9 enzyme inhibition (Veith)", ["non_inhibitor", "inhibitor"]),
    ("cyp2c19_veith", "CYP2C19 enzyme inhibition (Veith)", ["non_inhibitor", "inhibitor"]),
    ("cyp1a2_veith", "CYP1A2 enzyme inhibition (Veith)", ["non_inhibitor", "inhibitor"]),
    ("cyp2d6_substrate_carbonmangels", "CYP2D6 substrate specificity", ["non_substrate", "substrate"]),
    ("cyp3a4_substrate_carbonmangels", "CYP3A4 substrate specificity", ["non_substrate", "substrate"]),
    ("cyp2c9_substrate_carbonmangels", "CYP2C9 substrate specificity", ["non_substrate", "substrate"]),
    ("pgp_broccatelli", "P-glycoprotein transporter inhibition", ["non_inhibitor", "inhibitor"]),
    ("hia_hou", "Human intestinal absorption (Hou)", ["poor", "high"]),
    ("bioavailability_ma", "Oral drug bioavailability (Ma)", ["low", "high"]),
    ("bbb_martins", "Blood-brain barrier penetration (Martins)", ["low", "high"]),
    ("b3db_classification", "B3DB Blood-Brain Barrier distribution", ["non_permeable", "permeable"]),
    ("pampa_ncats", "PAMPA artificial membrane permeability", ["low", "high"]),
]

TDC_HTS_TASKS = [
    ("hiv", "HIV-1 replication inhibition in human T-cells", ["inactive", "active"]),
    ("sarscov2_3clpro_diamond", "SARS-CoV-2 3CLpro enzymatic inhibition", ["inactive", "active"]),
    ("kcnq2_potassium_channel_butkiewicz", "KCNQ2 potassium channel activation", ["inactive", "active"]),
    ("cav3_t-type_calcium_channels_butkiewicz", "Cav3 T-type calcium channel inhibition", ["inactive", "active"]),
]


def fetch_tdc_task(
    endpoint: str,
    group: str,
    desc: str,
    class_names: List[str],
    smoke: bool = False,
    smoke_samples: int = 60,
) -> TaskData:
    task_id = f"tdc_{endpoint}"

    try:
        from tdc.single_pred import ADME, Tox, HTS
        if group == "tox":
            loader_cls = Tox
        elif group == "adme":
            loader_cls = ADME
        else:
            loader_cls = HTS

        data = loader_cls(name=endpoint)
        split = data.get_split(method="scaffold", seed=42)
        train_df = split["train"]
        val_df = split["valid"]
        test_df = split["test"]
    except Exception as e:
        if not smoke:
            raise RuntimeError(f"Failed to fetch TDC {endpoint}: {e}") from e
        sample_smiles = [
            "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "c1ccccc1", "c1ccncc1", "CC(=O)NO", "O=C(O)c1ccccc1", "c1ccc2c(c1)cccc2",
            "CCO", "CCN", "CCC(=O)O", "CC(C)O", "CC#N", "Clc1ccccc1", "Fc1ccccc1",
        ] * 4
        n_samples = min(smoke_samples, len(sample_smiles))
        sample_smiles = sample_smiles[:n_samples]
        sample_labels = np.array([i % 2 for i in range(n_samples)], dtype=np.int64)
        n_tr = int(n_samples * 0.7)
        n_va = int(n_samples * 0.15)
        split_indices = SplitIndices(
            train=list(range(0, n_tr)),
            val=list(range(n_tr, n_tr + n_va)),
            test=list(range(n_tr + n_va, n_samples)),
        )
        meta = TaskMetadata(
            task_id=task_id,
            domain=Modality.MOLECULES.value,
            dataset_name=endpoint,
            num_classes=2,
            class_names=class_names,
            num_samples=n_samples,
            input_type="smiles",
            split_type="bemis_murcko_scaffold",
            split_sizes={"train": len(split_indices.train), "val": len(split_indices.val), "test": len(split_indices.test)},
            source=f"TDC {group.upper()}",
            description=f"{desc} (Smoke demo)",
        )
        return TaskData(metadata=meta, inputs=sample_smiles, labels=sample_labels, split_indices=split_indices)

    if smoke:
        train_df = train_df.iloc[: smoke_samples // 2]
        val_df = val_df.iloc[: smoke_samples // 4]
        test_df = test_df.iloc[: smoke_samples // 4]

    # Limit large tasks (like HIV with 40k samples) to 6000 max for fast and clean storage
    if len(train_df) > 4000:
        train_df = train_df.iloc[:4000]
    if len(val_df) > 1000:
        val_df = val_df.iloc[:1000]
    if len(test_df) > 1000:
        test_df = test_df.iloc[:1000]

    smiles_train = train_df["Drug"].astype(str).tolist()
    y_train = train_df["Y"].astype(int).to_numpy()

    smiles_val = val_df["Drug"].astype(str).tolist()
    y_val = val_df["Y"].astype(int).to_numpy()

    smiles_test = test_df["Drug"].astype(str).tolist()
    y_test = test_df["Y"].astype(int).to_numpy()

    all_inputs = smiles_train + smiles_val + smiles_test
    all_labels = np.concatenate([y_train, y_val, y_test]).astype(np.int64)

    n_tr, n_va, n_te = len(smiles_train), len(smiles_val), len(smiles_test)
    split_indices = SplitIndices(
        train=list(range(0, n_tr)),
        val=list(range(n_tr, n_tr + n_va)),
        test=list(range(n_tr + n_va, n_tr + n_va + n_te)),
    )

    metadata = TaskMetadata(
        task_id=task_id,
        domain=Modality.MOLECULES.value,
        dataset_name=endpoint,
        num_classes=2,
        class_names=class_names,
        num_samples=len(all_labels),
        input_type="smiles",
        split_type="bemis_murcko_scaffold",
        split_sizes={"train": n_tr, "val": n_va, "test": n_te},
        source=f"Therapeutics Data Commons ({group.upper()})",
        description=desc,
    )
    return TaskData(metadata=metadata, inputs=all_inputs, labels=all_labels, split_indices=split_indices)


def fetch_all_molecule_tasks(
    smoke: bool = False,
    limit: Optional[int] = None,
    store: Optional[Any] = None,
    force: bool = False,
) -> List[TaskData]:
    tasks = []
    task_specs = (
        [(ep, "tox", desc, cnames) for ep, desc, cnames in TDC_TOX_TASKS]
        + [(ep, "adme", desc, cnames) for ep, desc, cnames in TDC_ADME_TASKS]
        + [(ep, "hts", desc, cnames) for ep, desc, cnames in TDC_HTS_TASKS]
    )

    if limit is not None:
        task_specs = task_specs[:limit]

    for ep, group, desc, cnames in task_specs:
        task_id = f"tdc_{ep}"
        if store is not None and not force and store.has_task(Modality.MOLECULES.value, task_id):
            logger.info(f"Task '{task_id}' already exists in store, loading from disk...")
            try:
                tasks.append(store.load_task(Modality.MOLECULES.value, task_id))
                continue
            except Exception as e:
                logger.warning(f"Failed to load existing '{task_id}': {e}, re-fetching...")

        logger.info(f"Fetching molecule task: {ep} ({group})...")
        try:
            tdata = fetch_tdc_task(ep, group, desc, cnames, smoke=smoke)
            if store is not None:
                store.save_task(tdata)
            tasks.append(tdata)
        except Exception as e:
            logger.error(f"Failed to fetch molecule task {ep}: {e}")

    return tasks
