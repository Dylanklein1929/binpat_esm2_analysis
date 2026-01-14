#!/usr/bin/env python3
"""
03_compute_structural_metrics.py

Phase 1, Step 03:
- Compute DSSP-based structural metrics for predicted PDBs
- Write:
    (1) per-structure metrics CSV
    (2) per-template summary CSV (grouped by template_id)
    (3) skipped structures CSV

Inputs:
- --outdir : run directory (writes outputs here)
- --pdb-dir : directory containing predicted PDBs (default: <outdir>/pdbs)
- --variants-metadata : Step 01 variants_metadata.csv (maps variant_id -> template_id)

Notes:
- Requires external DSSP executable `mkdssp` on PATH.
- "avg. confidence" is mean_all_atom_bfactor reported in the PDB (raw values; no scaling).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from binpat.io.fasta import iter_fasta_records
from binpat.phase1.metrics import (
    MetricsSpec,
    SkippedStructure,
    compute_metrics_for_pdb,
    metrics_rows_to_dicts,
    skipped_rows_to_dicts,
    get_residues_to_skip_from_file,
)

from binpat.io.progress import Progress

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
    """
    Return 1-based positions covered by ALL occurrences of motif in seq.
    Overlapping matches allowed.
    Example: seq=AAAAAA, motif=AAA => matches at 1-3,2-4,3-5,4-6 => returns {1..6}
    """
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
        # i is 0-based start index in python string
        for pos0 in range(i, i + len(motif)):
            out.add(pos0 + 1)  # 1-based
        start = i + 1  # allow overlaps
    return out


def _load_variant_sequences(fasta_path: Optional[str]) -> Dict[str, str]:
    """
    Load FASTA into a dict: variant_id -> sequence.
    Uses iter_fasta_records(), which yields FastaRecord(id, seq, description).
    """
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
    """
    Returns mapping: variant_id -> group_column (e.g., template_id).
    If metadata is missing, returns empty dict (caller will use fallback heuristic).
    """
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
    """
    Fallback grouping when metadata is not provided:
    - seq1_var0003 -> seq1
    """
    if "_var" in variant_id:
        return variant_id.split("_var", 1)[0]
    return variant_id


def _write_csv(path: Path, rows: List[dict], *, empty_header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Write an empty file with header if provided
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
    """
    Build a per-group summary table with columns similar to your historical output.

    Output columns:
    - <group_col> (e.g., template_id)
    - sequence length
    - avg. confidence                             (mean_all_atom_bfactor mean; raw PDB values)
    - avg. helicity                               (helix_fraction mean)
    - avg. hydrophobic rASA                       (mean_hydrophobic_rasa mean)
    - fraction_with_rasa_below_threshold          (fraction with mean_hydrophobic_rasa <= threshold among non-missing rASA)
    - n_structures_total
    """
    df = per_structure_df.copy()

    # success only defined when mean_hydrophobic_rasa is present
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

    # rounding
    for col in ["avg. confidence", "avg. helicity", "avg. hydrophobic rASA", "fraction_with_rasa_below_threshold"]:
        out[col] = out[col].astype(float).round(3)

    # sequence length can be non-integer if mixed; keep one decimal
    out["sequence length"] = out["sequence length"].astype(float).round(1)

    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else (outdir / "pdbs")

    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    pdb_paths = sorted(pdb_dir.glob(args.pdb_glob))
    if not pdb_paths:
        raise ValueError(f"No PDB files found in {pdb_dir} matching glob '{args.pdb_glob}'")

    # ----------------------------
    # Build explicit skip set (1-based positions, as PDB residue numbers)
    # ----------------------------
    explicit_skip: Set[int] = set()
    if args.residues_to_skip is not None:
        explicit_skip = set(get_residues_to_skip_from_file(Path(args.residues_to_skip)))

    motifs: List[str] = args.skip_motif or []

    # Only load sequences if motifs requested
    variant_seqs: Dict[str, str] = {}
    if motifs:
        if args.variants_fasta is None:
            raise ValueError("--skip-motif was provided but --variants-fasta was not.")
        variant_seqs = _load_variant_sequences(args.variants_fasta)
        if not variant_seqs:
            raise ValueError(f"No sequences loaded from --variants-fasta: {args.variants_fasta}")

    print("[03_metrics] explicit skip positions (1-based):", sorted(explicit_skip))
    if motifs:
        print("[03_metrics] motif skip patterns:", motifs)
        print(f"[03_metrics] loaded sequences: {len(variant_seqs)}")

    # ----------------------------
    # Compute per-structure metrics (per-variant skip sets)
    # ----------------------------
    metric_rows = []
    skipped_rows = []

    p = Progress(total=len(pdb_paths), label="structures with metrics computed.")
    num = 0
    for pdb_path in pdb_paths:
        vid = pdb_path.stem

        skip_set: Set[int] = set(explicit_skip)

        # Add motif-derived skip positions
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

    # ----------------------------
    # Write outputs
    # ----------------------------
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
            "note",
        ],
    )
    _write_csv(
        skipped_path,
        skipped_rows_to_dicts(skipped_rows),
        empty_header=["variant_id", "pdb_path", "reason"],
    )

    # ----------------------------
    # Build summary
    # ----------------------------
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

        # sequence length proxy: DSSP residue count (already adjusted for skipping inside metrics)
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