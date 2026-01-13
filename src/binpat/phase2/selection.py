"""
selection.py

Helpers for selecting which variant_ids to embed in Phase 2.

Selection modes:
- all: embed all sequences from variants.fasta
- ids_file: embed only ids listed in a file (one variant_id per line)
- cluster_sample_k: for each (template_id, cluster_id) group, select up to k variants
  from cluster_assignments.csv (deterministic sorted order by default, or random with seed)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class SelectionSpec:
    mode: str  # "all" | "ids_file" | "cluster_sample_k"
    ids_file: Optional[str] = None

    # cluster_sample_k params
    cluster_assignments_csv: Optional[str] = None
    k_per_cluster: int = 5
    seed: int = 0
    sample: bool = False  # if False: take first k by sorted variant_id


def read_ids_file(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ids_file not found: {path}")
    out: Set[str] = set()
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def select_variant_ids(
    all_variant_ids: Sequence[str],
    *,
    spec: SelectionSpec,
) -> List[str]:
    """
    Returns an ordered list of selected variant_ids.
    Order is stable/deterministic unless spec.sample=True.
    """
    mode = (spec.mode or "").lower().strip()
    if mode not in ("all", "ids_file", "cluster_sample_k"):
        raise ValueError(f"Unknown selection mode: {spec.mode!r}")

    # All
    if mode == "all":
        return list(all_variant_ids)

    # ids_file
    if mode == "ids_file":
        if not spec.ids_file:
            raise ValueError("mode=ids_file requires ids_file")
        keep = read_ids_file(spec.ids_file)
        return [vid for vid in all_variant_ids if vid in keep]

    # cluster_sample_k
    if mode == "cluster_sample_k":
        if not spec.cluster_assignments_csv:
            raise ValueError("mode=cluster_sample_k requires cluster_assignments_csv")
        if spec.k_per_cluster <= 0:
            raise ValueError("k_per_cluster must be > 0")

        df = pd.read_csv(spec.cluster_assignments_csv)
        required = {"variant_id", "template_id", "cluster_id"}
        missing_cols = required.difference(set(df.columns))
        if missing_cols:
            raise ValueError(
                "cluster_assignments.csv is missing required columns: "
                + ", ".join(sorted(missing_cols))
            )

        # Only keep variant_ids that are present in fasta (all_variant_ids)
        in_fasta = set(all_variant_ids)
        df = df[df["variant_id"].astype(str).isin(in_fasta)].copy()

        # Drop missing cluster_id rows
        df = df[df["cluster_id"].notnull()].copy()

        # Group by (template_id, cluster_id)
        rng = random.Random(int(spec.seed))
        selected: List[str] = []

        for (template_id, cluster_id), sub in df.groupby(["template_id", "cluster_id"]):
            vids = [str(v) for v in sub["variant_id"].tolist()]
            vids_sorted = sorted(vids)
            if spec.sample:
                if len(vids_sorted) <= spec.k_per_cluster:
                    chosen = vids_sorted
                else:
                    chosen = rng.sample(vids_sorted, spec.k_per_cluster)
                    chosen = sorted(chosen)  # keep stable order in output
            else:
                chosen = vids_sorted[: spec.k_per_cluster]
            selected.extend(chosen)

        # preserve original fasta order as a final stable tie-breaker:
        selected_set = set(selected)
        return [vid for vid in all_variant_ids if vid in selected_set]

    # unreachable
    return list(all_variant_ids)
