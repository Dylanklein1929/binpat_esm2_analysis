#!/usr/bin/env python3
"""
03_compute_structural_metrics.py

Phase 1, Step 03:
- Compute DSSP-based structural metrics for predicted PDBs
- Write:
    (1) per-structure metrics CSV
    (2) per-template summary CSV (grouped by template_id)
    (3) skipped structures CSV

Topology gate behavior:
- All structures remain included in the ordinary metric averages
- fraction_with_rasa_below_threshold uses rasa_success_after_topology_gate
- The centroid-based topology check is applied only to structures that already
  satisfy the raw hydrophobic rASA threshold
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from binpat.io.fasta import iter_fasta_records
from binpat.io.progress import Progress
from binpat.phase1.metrics import (
    MetricsSpec,
    SkippedStructure,
    compute_metrics_for_pdb,
    metrics_rows_to_dicts,
    skipped_rows_to_dicts,
    get_residues_to_skip_from_file,
)

# Topology-gate defaults.
#
# Helix ranges are derived separately for each structure from template
# information embedded in variant_id, for example:
#
#   binpat_len_100|helices|21,20,20,21|loops|6,6,6|...
#
# By default, the central eight residues of each encoded helix are used to
# calculate its centroid.
DEFAULT_TOPOLOGY_HELIX_WINDOW_SIZE: int = 8
DEFAULT_CENTROID_ABS_COSINE_THRESHOLD: float = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute structural metrics from predicted PDBs.")
    p.add_argument("--outdir", required=True, type=str, help="Run directory (writes CSVs here).")
    p.add_argument("--pdb-dir", type=str, default=None, help="Directory of PDBs (default: <outdir>/pdbs).")
    p.add_argument("--pdb-glob", type=str, default="*.pdb", help="Glob pattern inside pdb-dir (default: *.pdb).")

    p.add_argument(
        "--variants-metadata",
        type=str,
        default=None,
        help="Path to Step 01 variants_metadata.csv (used to map variant_id -> template_id).",
    )
    p.add_argument(
        "--group-column",
        type=str,
        default="template_id",
        help="Column name in variants-metadata to use for grouping (default: template_id).",
    )

    p.add_argument(
        "--apply-top-check-with-rasa-gate",
        "--apply_top_check_with_rasa_gate",
        dest="apply_top_check_with_rasa_gate",
        action="store_true",
        help=(
            "Apply the centroid-connectivity topology gate only after a "
            "structure passes the hydrophobic-rASA threshold."
        ),
    )
    p.add_argument(
        "--topology-helix-window-size",
        type=int,
        default=DEFAULT_TOPOLOGY_HELIX_WINDOW_SIZE,
        help=(
            "Number of central residues from each template-defined helix to "
            "use for its centroid (default: 8)."
        ),
    )
    p.add_argument(
        "--use-full-topology-helices",
        action="store_true",
        help=(
            "Use each full template-defined helix for its centroid instead "
            "of a fixed-width central window."
        ),
    )
    p.add_argument(
        "--topology-abs-cosine-threshold",
        type=float,
        default=DEFAULT_CENTROID_ABS_COSINE_THRESHOLD,
        help=(
            "Minimum absolute cosine similarity required to pass the "
            "topology gate (default: 0.2)."
        ),
    )

    p.add_argument("--rasa-threshold", type=float, default=0.25, help="Success threshold for mean hydrophobic rASA.")
    p.add_argument("--model-index", type=int, default=0, help="Model index for multi-model PDBs (default: 0).")

    p.add_argument(
        "--residues-to-skip",
        type=str,
        default=None,
        help="Path to file containing comma-separated list of residue indices to omit from structure metric calculation (1-based)."
    )

    p.add_argument(
        "--variants-fasta",
        type=str,
        default=None,
        help="FASTA mapping variant_id -> sequence (headers must match variant_id). Required if using --skip-motif.",
    )

    p.add_argument(
        "--skip-motif",
        action="append",
        default=[],
        help="Motif to omit from metrics (repeatable). Example: --skip-motif GGGGG",
    )

    return p.parse_args()


def _motif_positions_1based(seq: str, motif: str) -> Set[int]:
    seq = seq.upper()
    motif = motif.upper()
    if not motif:
        return set()

    out: Set[int] = set()
    start = 0
    while True:
        i = seq.find(motif, start)
        if i == -1:
            break
        for pos0 in range(i, i + len(motif)):
            out.add(pos0 + 1)
        start = i + 1
    return out


def _load_variant_sequences(fasta_path: Optional[str]) -> Dict[str, str]:
    if fasta_path is None:
        return {}
    seqs: Dict[str, str] = {}
    for rec in iter_fasta_records(Path(fasta_path)):
        seqs[str(rec.id)] = str(rec.seq)
    return seqs


def _read_variant_to_group_map(
    variants_metadata_csv: Optional[str],
    group_column: str,
) -> Dict[str, str]:
    if not variants_metadata_csv:
        return {}

    path = Path(variants_metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"--variants-metadata not found: {path}")

    df = pd.read_csv(path)
    if "variant_id" not in df.columns:
        raise ValueError(
            f"--variants-metadata must contain a 'variant_id' column. Found: {list(df.columns)}"
        )
    if group_column not in df.columns:
        raise ValueError(
            f"--group-column '{group_column}' not found in variants metadata. Found: {list(df.columns)}"
        )

    return {str(r["variant_id"]): str(r[group_column]) for _, r in df.iterrows()}


def _default_group_from_variant_id(variant_id: str) -> str:
    if "_var" in variant_id:
        return variant_id.split("_var", 1)[0]
    return variant_id


def _write_csv(path: Path, rows: List[dict], *, empty_header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        if empty_header is not None:
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(empty_header)
        else:
            path.write_text("")
        return

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_summary_table(
    per_structure_df: pd.DataFrame,
    *,
    group_col: str,
    rasa_threshold: float,
) -> pd.DataFrame:
    df = per_structure_df.copy()

    # Prefer the topology-gated success flag when present
    if "rasa_success_after_topology_gate" in df.columns:
        def normalize_success(x):
            if pd.isna(x):
                return None
            return bool(int(x))
        df["is_success"] = df["rasa_success_after_topology_gate"].apply(normalize_success)
    else:
        def to_success(x):
            if pd.isna(x):
                return None
            return float(x) <= float(rasa_threshold)
        df["is_success"] = df["mean_hydrophobic_rasa"].apply(to_success)

    def success_rate(series: pd.Series) -> float:
        vals = [v for v in series.tolist() if v is not None]
        if not vals:
            return float("nan")
        return sum(bool(v) for v in vals) / len(vals)

    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            **{
                "sequence length": ("sequence_length", "mean"),
                "avg. confidence": ("mean_all_atom_bfactor", "mean"),
                "avg. helicity": ("helix_fraction", "mean"),
                "avg. hydrophobic rASA": ("mean_hydrophobic_rasa", "mean"),
                "fraction_with_rasa_below_threshold": ("is_success", success_rate),
                "n_structures_total": ("variant_id", "count"),
            }
        )
        .reset_index()
    )

    for col in ["avg. confidence", "avg. helicity", "avg. hydrophobic rASA", "fraction_with_rasa_below_threshold"]:
        out[col] = out[col].astype(float).round(3)

    out["sequence length"] = out["sequence length"].astype(float).round(1)
    return out


def main() -> None:
    args = parse_args()

    if args.topology_helix_window_size <= 0:
        raise ValueError("--topology-helix-window-size must be a positive integer.")
    if not 0.0 <= args.topology_abs_cosine_threshold <= 1.0:
        raise ValueError("--topology-abs-cosine-threshold must be between 0 and 1.")

    topology_window_size: Optional[int]
    if args.use_full_topology_helices:
        topology_window_size = None
    else:
        topology_window_size = int(args.topology_helix_window_size)

    outdir = Path(args.outdir)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else (outdir / "pdbs")

    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    pdb_paths = sorted(pdb_dir.glob(args.pdb_glob))
    if not pdb_paths:
        raise ValueError(f"No PDB files found in {pdb_dir} matching glob '{args.pdb_glob}'")

    explicit_skip: Set[int] = set()
    if args.residues_to_skip is not None:
        explicit_skip = set(get_residues_to_skip_from_file(Path(args.residues_to_skip)))

    motifs: List[str] = args.skip_motif or []

    variant_seqs: Dict[str, str] = {}
    if motifs:
        if args.variants_fasta is None:
            raise ValueError("--skip-motif was provided but --variants-fasta was not.")
        variant_seqs = _load_variant_sequences(args.variants_fasta)
        if not variant_seqs:
            raise ValueError(f"No sequences loaded from --variants-fasta: {args.variants_fasta}")

    print("[03_metrics] explicit skip positions (1-based):", sorted(explicit_skip))
    if args.apply_top_check_with_rasa_gate:
        window_description = (
            "full template-defined helices"
            if topology_window_size is None
            else f"central {topology_window_size} residues per template-defined helix"
        )
        print("[03_metrics] topology gate: enabled")
        print("[03_metrics] topology helix ranges:", window_description)
        print(
            "[03_metrics] topology abs-cosine threshold:",
            float(args.topology_abs_cosine_threshold),
        )
    else:
        print("[03_metrics] topology gate: disabled")

    if motifs:
        print("[03_metrics] motif skip patterns:", motifs)
        print(f"[03_metrics] loaded sequences: {len(variant_seqs)}")

    metric_rows = []
    skipped_rows = []

    p = Progress(total=len(pdb_paths), label="structures with metrics computed.")
    num = 0
    for pdb_path in pdb_paths:
        vid = pdb_path.stem

        skip_set: Set[int] = set(explicit_skip)

        if motifs:
            seq = variant_seqs.get(vid)
            if seq is None:
                skipped_rows.append(SkippedStructure(
                    variant_id=vid,
                    pdb_path=str(pdb_path),
                    reason="Missing sequence for variant_id in --variants-fasta (needed for --skip-motif).",
                ))
                continue

            for m in motifs:
                skip_set |= _motif_positions_1based(seq, m)

        spec = MetricsSpec(
            rasa_threshold=float(args.rasa_threshold),
            model_index=int(args.model_index),
            residues_to_skip=skip_set,
            apply_topology_check_with_rasa_gate=bool(
                args.apply_top_check_with_rasa_gate
            ),
            derive_helix_ranges_from_variant_id=True,
            helix_ranges=None,
            topology_helix_window_size=topology_window_size,
            chain_id=None,
            abs_cosine_threshold=float(
                args.topology_abs_cosine_threshold
            ),
        )

        try:
            metric_rows.append(compute_metrics_for_pdb(pdb_path, spec=spec, variant_id=vid))
            p.update(num, extra=vid)
            num += 1
        except Exception as e:
            skipped_rows.append(SkippedStructure(
                variant_id=vid,
                pdb_path=str(pdb_path),
                reason=f"{type(e).__name__}: {e}",
            ))

    per_structure_path = outdir / "structural_metrics_per_structure.csv"
    summary_path = outdir / "structural_metrics_summary.csv"
    skipped_path = outdir / "skipped_structures.csv"

    _write_csv(
        per_structure_path,
        metrics_rows_to_dicts(metric_rows),
        empty_header=[
            "variant_id",
            "pdb_path",
            "helix_fraction",
            "mean_hydrophobic_rasa",
            "mean_all_atom_bfactor",
            "n_res_dssp",
            "n_helix_res",
            "n_hydrophobic_res",
            "mean_hydrophobic_rasa_leq_threshold",
            "topology_checked_for_rasa_success",
            "bad_topology_among_rasa_passers",
            "rasa_success_after_topology_gate",
            "topology_abs_cosine_similarity",
            "topology_folded_angle_degrees",
            "topology_failure_reason",
            "topology_helix_ranges",
            "topology_helix_range_source",
            "note",
        ],
    )
    _write_csv(
        skipped_path,
        skipped_rows_to_dicts(skipped_rows),
        empty_header=["variant_id", "pdb_path", "reason"],
    )

    vid_to_group = _read_variant_to_group_map(args.variants_metadata, args.group_column)

    df = pd.read_csv(per_structure_path)
    if df.empty:
        _write_csv(
            summary_path,
            [],
            empty_header=[
                args.group_column,
                "sequence length",
                "avg. confidence",
                "avg. helicity",
                "avg. hydrophobic rASA",
                "fraction_with_rasa_below_threshold",
                "n_structures_total",
            ],
        )
    else:
        df[args.group_column] = df["variant_id"].apply(
            lambda vid: vid_to_group.get(str(vid), _default_group_from_variant_id(str(vid)))
        )
        df["sequence_length"] = df["n_res_dssp"].astype(float)

        summary_df = _make_summary_table(
            df,
            group_col=args.group_column,
            rasa_threshold=float(args.rasa_threshold),
        )
        summary_df.to_csv(summary_path, index=False)

    print(f"[03_metrics] pdb_dir: {pdb_dir}  (n_pdb={len(pdb_paths)})")
    print(f"[03_metrics] ok: {len(metric_rows)}  skipped: {len(skipped_rows)}")
    print(f"[03_metrics] wrote: {per_structure_path}")
    print(f"[03_metrics] wrote: {summary_path}")
    print(f"[03_metrics] wrote: {skipped_path}")


if __name__ == "__main__":
    main()
