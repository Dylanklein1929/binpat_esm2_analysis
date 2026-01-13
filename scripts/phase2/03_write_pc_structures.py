#!/usr/bin/env python3
"""
03_write_pc_structures.py

Phase 2, Step 03:
- Take fitted PCA model + embeddings (or precomputed pca_scores)
- Write PCk values into B-factor column for each variant's predicted PDB

Defaults:
- Reads PDBs from: <outdir>/pdbs/<variant_id>.pdb
- Writes to:        <outdir>/pdbs_pc/pc<k>/<variant_id>_pc<k>.pdb

Inputs:
- PCA model: <outdir>/phase2/pca/pca_model.joblib
- Either:
  A) per-variant pca_scores: <outdir>/phase2/pca_scores/<variant_id>.npz (key 'pc')
  or
  B) embeddings: <outdir>/phase2/embeddings/<variant_id>.npz (key 'emb'), then project on the fly.

Usage:
  python scripts/phase2/03_write_pc_structures.py \
    --outdir results/run001 \
    --which-pc 1 \
    --use-precomputed-scores
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from joblib import load

from binpat.phase2.pca import project_one
from binpat.phase2.pc_structures import PCWriteResult, insert_pc_into_pdb_bfactor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write PC values into PDB B-factors for visualization.")
    p.add_argument("--outdir", required=True, type=str)

    p.add_argument("--which-pc", required=True, type=int, help="1-based PC index (1=PC1)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])

    p.add_argument("--pdb-dir", type=str, default=None, help="Default: <outdir>/pdbs")
    p.add_argument("--out-pdb-root", type=str, default=None, help="Default: <outdir>/pdbs_pc")

    p.add_argument("--pca-model", type=str, default=None, help="Default: <outdir>/phase2/pca/pca_model.joblib")
    p.add_argument("--embeddings-dir", type=str, default=None, help="Default: <outdir>/phase2/embeddings")
    p.add_argument("--scores-dir", type=str, default=None, help="Default: <outdir>/phase2/pca_scores")

    # selection
    p.add_argument("--ids-file", type=str, default=None, help="Optional: file with one variant_id per line.")
    p.add_argument("--limit", type=int, default=None)

    # behavior
    p.add_argument("--use-precomputed-scores", action="store_true", help="Use pca_scores/<id>.npz if present.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _read_ids_file(path: str) -> List[str]:
    ids: List[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def _select_ids(default_dir: Path, *, ids_file: Optional[str], limit: Optional[int]) -> List[str]:
    if ids_file:
        ids = _read_ids_file(ids_file)
    else:
        # fallback: infer from embeddings dir or scores dir presence
        ids = [p.stem for p in sorted(default_dir.glob("*.npz"))]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        ids = ids[:limit]
    return ids


def write_report_csv(path: Path, rows: List[PCWriteResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "in_pdb", "out_pdb", "ok", "reason", "n_ca_written", "n_pc_available"])
        for r in rows:
            w.writerow([r.variant_id, r.in_pdb, r.out_pdb, r.ok, r.reason or "", r.n_ca_written, r.n_pc_available])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    which_pc = int(args.which_pc)
    if which_pc < 1:
        raise ValueError("--which-pc must be >= 1 (1-based)")

    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else (outdir / "pdbs")
    out_root = Path(args.out_pdb_root) if args.out_pdb_root else (outdir / "pdbs_pc")

    pca_model_path = Path(args.pca_model) if args.pca_model else (outdir / "phase2" / "pca" / "pca_model.joblib")
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else (outdir / "phase2" / "embeddings")
    scores_dir = Path(args.scores_dir) if args.scores_dir else (outdir / "phase2" / "pca_scores")

    bundle = load(pca_model_path)
    pca = bundle["pca"]

    # choose a default NPZ dir to infer IDs from
    infer_dir = scores_dir if args.use_precomputed_scores and scores_dir.exists() else embeddings_dir
    ids = _select_ids(infer_dir, ids_file=args.ids_file, limit=args.limit)
    if not ids:
        raise ValueError("No variant IDs selected.")

    out_pc_dir = out_root / f"pc{which_pc}"
    report_csv = out_pc_dir / "pc_write_report.csv"

    rows: List[PCWriteResult] = []

    for vid in ids:
        in_pdb = pdb_dir / f"{vid}.pdb"
        if not in_pdb.exists():
            rows.append(PCWriteResult(vid, str(in_pdb), "", False, "missing_pdb", 0, 0))
            continue

        out_pdb = out_pc_dir / f"{vid}_pc{which_pc}.pdb"
        if out_pdb.exists() and not args.overwrite:
            rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), True, "skipped_exists", 0, 0))
            continue

        # get pc scores
        pc_scores = None

        if args.use_precomputed_scores:
            score_path = scores_dir / f"{vid}.npz"
            if score_path.exists():
                with np.load(score_path, allow_pickle=False) as z:
                    if "pc" not in z:
                        rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), False, "bad_scores_npz_missing_pc", 0, 0))
                        continue
                    pc_scores = z["pc"]
            # else: fall back to embeddings

        if pc_scores is None:
            emb_path = embeddings_dir / f"{vid}.npz"
            if not emb_path.exists():
                rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), False, "missing_embeddings_and_scores", 0, 0))
                continue
            with np.load(emb_path, allow_pickle=False) as z:
                if "emb" not in z:
                    rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), False, "bad_embeddings_npz_missing_emb", 0, 0))
                    continue
                emb = z["emb"]
            pc_scores = project_one(pca, emb, dtype=str(args.dtype))

        if pc_scores.ndim != 2 or pc_scores.shape[1] < which_pc:
            rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), False, "pc_scores_shape_mismatch", 0, int(pc_scores.shape[1]) if pc_scores.ndim == 2 else 0))
            continue

        pc_vals = pc_scores[:, which_pc - 1]
        n_written, n_pc, note = insert_pc_into_pdb_bfactor(in_pdb, out_pdb, pc_vals)

        ok = True
        reason = note  # mismatch note if any
        rows.append(PCWriteResult(vid, str(in_pdb), str(out_pdb), ok, reason, n_written, n_pc))

    write_report_csv(report_csv, rows)

    n_ok = sum(1 for r in rows if r.ok and (r.reason or "") != "skipped_exists")
    n_skip = sum(1 for r in rows if r.ok and (r.reason or "") == "skipped_exists")
    n_fail = sum(1 for r in rows if not r.ok)

    print(f"[phase2/03_write_pc_structures] selected: {len(rows)} ok: {n_ok} skipped: {n_skip} failed: {n_fail}")
    print(f"[phase2/03_write_pc_structures] out: {out_pc_dir}")
    print(f"[phase2/03_write_pc_structures] report: {report_csv}")


if __name__ == "__main__":
    main()
