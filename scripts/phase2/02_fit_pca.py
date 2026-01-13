#!/usr/bin/env python3
"""
02_fit_pca.py

Phase 2, Step 02:
- Fit PCA (IncrementalPCA) on residue embeddings saved by Phase2 Step01.
- Optionally write per-variant PC scores into outdir/phase2/pca_scores/<id>.npz

Writes:
- outdir/phase2/pca/pca_model.joblib
- outdir/phase2/pca/pca_report.csv
- (optional) outdir/phase2/pca_scores/<variant_id>.npz  (key 'pc': (L, n_components))

Usage:
  python scripts/phase2/02_fit_pca.py \
    --outdir results/run001 \
    --embeddings-dir results/run001/phase2/embeddings \
    --model facebook/esm2_t33_650M_UR50D \
    --n-components 10 \
    --write-scores
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from joblib import load

from binpat.phase2.pca import (
    PCASpec,
    fit_incremental_pca,
    project_one,
    write_pca_artifacts,
    write_scores_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit IncrementalPCA on ESM2 residue embeddings.")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--embeddings-dir", required=True, type=str)
    p.add_argument("--model", required=True, type=str, help="ESM2 model name used in Step01 (for bookkeeping).")

    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--batch-residues", type=int, default=200_000)
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    p.add_argument("--max-residues-total", type=int, default=None)

    p.add_argument("--emb-key", type=str, default="emb", help="NPZ key for embeddings array (L,D).")

    # selection
    p.add_argument("--ids-file", type=str, default=None, help="Optional: file with one variant_id per line.")
    p.add_argument("--limit", type=int, default=None, help="Optional: only first N IDs (after ids-file filtering).")

    # outputs
    p.add_argument("--write-scores", action="store_true", help="Also write per-variant PC scores.")
    return p.parse_args()


def _read_ids_file(path: str) -> List[str]:
    ids: List[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def _select_ids(emb_dir: Path, *, ids_file: Optional[str], limit: Optional[int]) -> Optional[List[str]]:
    if ids_file is None:
        ids = None
    else:
        ids = _read_ids_file(ids_file)

    if ids is not None and limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        ids = ids[:limit]
    return ids


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    emb_dir = Path(args.embeddings_dir)

    ids = _select_ids(emb_dir, ids_file=args.ids_file, limit=args.limit)

    spec = PCASpec(
        n_components=int(args.n_components),
        batch_residues=int(args.batch_residues),
        dtype=str(args.dtype),
        max_residues_total=int(args.max_residues_total) if args.max_residues_total is not None else None,
    )

    pca, report = fit_incremental_pca(
        emb_dir,
        spec=spec,
        ids=ids,
        emb_key=str(args.emb_key),
    )

    pca_dir = outdir / "phase2" / "pca"
    artifacts = write_pca_artifacts(pca_dir, pca=pca, report=report, model_name=str(args.model))

    # optionally write scores
    if args.write_scores:
        scores_dir = outdir / "phase2" / "pca_scores"
        # decide which IDs to score: if ids is None, score all present in emb_dir
        if ids is None:
            ids_to_score = [p.stem for p in sorted(emb_dir.glob("*.npz"))]
        else:
            ids_to_score = ids

        for vid in ids_to_score:
            npz_path = emb_dir / f"{vid}.npz"
            with np.load(npz_path, allow_pickle=False) as z:
                emb = z[str(args.emb_key)]
            pc_scores = project_one(pca, emb, dtype=str(args.dtype))
            write_scores_npz(scores_dir, vid, pc_scores)

    print(f"[phase2/02_fit_pca] wrote model:  {artifacts['model']}")
    print(f"[phase2/02_fit_pca] wrote report: {artifacts['report']}")
    if args.write_scores:
        print(f"[phase2/02_fit_pca] wrote per-variant scores: {(outdir / 'phase2' / 'pca_scores')}")


if __name__ == "__main__":
    main()
