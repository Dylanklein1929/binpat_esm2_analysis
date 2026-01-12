#!/usr/bin/env python3
"""
05_average_structures.py

Phase 1, Step 05:
- For each template_id and each cluster, compute an average structure.
- Align all cluster members to the cluster medoid, then average backbone coords.

Inputs:
- outdir/clusters/<template_id>/cluster_assignments.csv
  (must contain: variant_id, cluster_id, is_medoid)
- PDBs in outdir/pdbs/<variant_id>.pdb

Outputs:
- outdir/average_structures/<template_id>/cluster<cluster_id>_avg.pdb
- outdir/average_structures_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from binpat.phase1.averaging import AverageSpec, average_cluster_structures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute average structures per cluster (aligned to medoids).")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--pdb-dir", type=str, default=None, help="Default: <outdir>/pdbs")
    p.add_argument("--clusters-dir", type=str, default=None, help="Default: <outdir>/clusters")
    p.add_argument("--avg-outdir", type=str, default=None, help="Default: <outdir>/average_structures")

    p.add_argument("--align-atoms", type=str, default="CA", help="Comma-separated atom names for alignment, e.g. CA or N,CA,C")
    p.add_argument("--avg-atoms", type=str, default="N,CA,C", help="Comma-separated atom names to average/write")
    p.add_argument("--no-require-same-length", action="store_true", help="Allow mismatched atom counts (not recommended).")

    return p.parse_args()


def _split_atoms(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else (outdir / "pdbs")
    clusters_dir = Path(args.clusters_dir) if args.clusters_dir else (outdir / "clusters")
    avg_outdir = Path(args.avg_outdir) if args.avg_outdir else (outdir / "average_structures")
    avg_outdir.mkdir(parents=True, exist_ok=True)

    spec = AverageSpec(
        align_atom_names=tuple(_split_atoms(args.align_atoms)),
        avg_atom_names=tuple(_split_atoms(args.avg_atoms)),
        require_same_length=not bool(args.no_require_same_length),
        quiet=True,
    )

    manifest_rows: List[dict] = []

    # Each template has its own folder under clusters/
    if not clusters_dir.exists():
        raise FileNotFoundError(f"clusters_dir does not exist: {clusters_dir}")

    template_dirs = [p for p in clusters_dir.iterdir() if p.is_dir()]
    if not template_dirs:
        raise ValueError(f"No template directories found under {clusters_dir}")

    for tdir in sorted(template_dirs):
        template_id = tdir.name
        assign_csv = tdir / "cluster_assignments.csv"
        if not assign_csv.exists():
            continue

        df = pd.read_csv(assign_csv)
        if not {"variant_id", "cluster_id", "is_medoid"}.issubset(df.columns):
            raise ValueError(f"{assign_csv} missing required columns. Found: {list(df.columns)}")

        for cluster_id, df_c in df.groupby("cluster_id"):
            # skip NaN cluster_id rows (missing pdb rows etc.)
            if pd.isna(cluster_id):
                continue

            cluster_id_int = int(cluster_id)
            members = [str(v) for v in df_c["variant_id"].tolist()]

            # find medoid
            med_df = df_c[df_c["is_medoid"] == True]
            if med_df.empty:
                # fallback: first member
                ref_vid = members[0]
            else:
                ref_vid = str(med_df.iloc[0]["variant_id"])

            out_pdb = avg_outdir / template_id / f"cluster{cluster_id_int:03d}_avg.pdb"

            try:
                res = average_cluster_structures(
                    template_id=template_id,
                    cluster_id=cluster_id_int,
                    member_variant_ids=members,
                    pdb_dir=pdb_dir,
                    out_pdb=out_pdb,
                    reference_variant_id=ref_vid,
                    spec=spec,
                )
                manifest_rows.append(
                    {
                        "template_id": res.template_id,
                        "cluster_id": res.cluster_id,
                        "reference_variant_id": res.reference_variant_id,
                        "n_total": res.n_total,
                        "n_used": res.n_used,
                        "n_skipped": res.n_skipped,
                        "out_pdb_path": res.out_pdb_path,
                        "skipped_variant_ids": ";".join(res.skipped_variant_ids),
                        "note": res.note,
                    }
                )
            except Exception as e:
                manifest_rows.append(
                    {
                        "template_id": template_id,
                        "cluster_id": cluster_id_int,
                        "reference_variant_id": ref_vid,
                        "n_total": len(members),
                        "n_used": 0,
                        "n_skipped": len(members),
                        "out_pdb_path": "",
                        "skipped_variant_ids": ";".join(members),
                        "note": f"average_failed: {type(e).__name__}: {e}",
                    }
                )

    manifest_path = outdir / "average_structures_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"[05_average] wrote: {manifest_path}")
    print(f"[05_average] avg_outdir: {avg_outdir}")


if __name__ == "__main__":
    main()
