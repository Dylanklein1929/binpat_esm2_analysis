#!/usr/bin/env python3
"""
04_cluster_structures.py

Phase 1, Step 04:
- Cluster predicted structures by CA RMSD (hierarchical clustering) per template_id.

Writes:
- outdir/cluster_assignments.csv
- outdir/cluster_medoids.csv
- outdir/clusters/<template_id>/cluster_assignments.csv
- outdir/clusters/<template_id>/cluster_medoids.csv
- outdir/clusters/<template_id>/cluster_summary.txt
- outdir/clusters/<template_id>/dendrogram.(png|pdf|svg)  [optional]

Notes:
- O(N^2) RMSD computation per template_id (pairwise RMSD matrix). Could improve this later with caching.

Example:
python 04_cluster_structures.py \
    --outdir outdir/ \
    --variants-metadata variants_metadata.csv \
    --linkage single \
    ==dendrogram

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from binpat.phase1.rmsd_clustering import (
    RMSDClusterSpec,
    candidate_cutoffs_from_rmsd_matrix,
    choose_cutoff_by_silhouette,
    cluster_labels_from_linkage,
    compute_cluster_medoids,
    compute_linkage_from_rmsd,
    compute_rmsd_matrix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster structures by CA RMSD per template.")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--variants-metadata", required=True, type=str, help="variants_metadata.csv from Step 01")
    p.add_argument("--pdb-dir", type=str, default=None, help="Default: <outdir>/pdbs")

    p.add_argument("--atom-name", type=str, default="CA")
    p.add_argument("--linkage", type=str, default="single", choices=["single", "complete", "average"])
    p.add_argument("--fixed-cutoff", type=float, default=None, help="Skip silhouette search and use this cutoff.")
    p.add_argument("--n-cutoffs", type=int, default=50)
    p.add_argument("--min-clusters", type=int, default=2)
    p.add_argument("--max-clusters", type=int, default=20)

    p.add_argument("--dendrogram", action="store_true", help="Write dendrogram image per template.")
    p.add_argument("--dendrogram-format", type=str, default="png", choices=["png", "pdf", "svg"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else (outdir / "pdbs")

    meta = pd.read_csv(args.variants_metadata)
    if "variant_id" not in meta.columns or "template_id" not in meta.columns:
        raise ValueError(
            f"variants_metadata must contain variant_id and template_id. Found: {list(meta.columns)}"
        )

    rows_all: List[dict] = []
    medoids_all: List[dict] = []

    for template_id, df_t in meta.groupby("template_id"):
        variant_ids = [str(v) for v in df_t["variant_id"].tolist()]
        pdb_paths: Dict[str, Path] = {}
        missing: List[str] = []

        for vid in variant_ids:
            p = pdb_dir / f"{vid}.pdb"
            if p.exists():
                pdb_paths[vid] = p
            else:
                missing.append(vid)

        # require at least 2 structures to cluster
        if len(pdb_paths) < 2:
            for vid in variant_ids:
                rows_all.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": None,
                        "is_medoid": None,
                        "cutoff": None,
                        "silhouette": None,
                        "n_clusters": None,
                        "note": "missing_pdb_or_too_few_structures" if vid in missing else "too_few_structures",
                    }
                )
            continue

        # remove structures with criss-crossing glycine loops
        #valid_pdb_paths, metrics_by_id = filter_structs_by_centroid_connectivity(pdb_paths,
        
        spec = RMSDClusterSpec(
            atom_name=args.atom_name,
            linkage_method=args.linkage,
            fixed_cutoff=args.fixed_cutoff,
            n_cutoffs=int(args.n_cutoffs),
            min_clusters=int(args.min_clusters),
            max_clusters=int(args.max_clusters),
        )

        # --- Compute + cluster inside try/except (for reporting failures cleanly) ---
        try:
            ids, rmsd_mat = compute_rmsd_matrix(pdb_paths, atom_name=spec.atom_name)
            Z = compute_linkage_from_rmsd(rmsd_mat, method=spec.linkage_method)

            if spec.fixed_cutoff is not None:
                cutoff = float(spec.fixed_cutoff)
                labels = cluster_labels_from_linkage(Z, cutoff=cutoff, criterion=spec.criterion)
                sil = None
            else:
                cutoffs = candidate_cutoffs_from_rmsd_matrix(
                    rmsd_mat,
                    n_cutoffs=spec.n_cutoffs,
                    qmin=spec.cutoff_min_quantile,
                    qmax=spec.cutoff_max_quantile,
                )
                cutoff, sil, labels = choose_cutoff_by_silhouette(
                    rmsd_mat,
                    Z,
                    criterion=spec.criterion,
                    cutoffs=cutoffs,
                    min_clusters=spec.min_clusters,
                    max_clusters=spec.max_clusters,
                )

            n_clusters = len(set(labels))
            medoids = compute_cluster_medoids(ids, rmsd_mat, labels)  # cluster_id -> variant_id

            # per-template output dir
            tdir = outdir / "clusters" / str(template_id)
            tdir.mkdir(parents=True, exist_ok=True)

            # optional dendrogram
            if args.dendrogram:
                fig = plt.figure()
                dendrogram(Z, labels=ids, orientation="top")
                fig.tight_layout()
                fig.savefig(tdir / f"dendrogram.{args.dendrogram_format}", dpi=300)
                plt.close(fig)

            # Write per-structure assignments
            per_rows: List[dict] = []
            for vid, lab in zip(ids, labels):
                per_rows.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": int(lab),
                        "is_medoid": (vid == medoids[int(lab)]), # later used as reference structure for computing average structure
                        "cutoff": cutoff,
                        "silhouette": sil,
                        "n_clusters": n_clusters,
                        "note": "",
                    }
                )
            pd.DataFrame(per_rows).to_csv(tdir / "cluster_assignments.csv", index=False)

            # Write per-template medoids table
            medoid_rows = [
                {"template_id": template_id, "cluster_id": int(cid), "medoid_variant_id": mid}
                for cid, mid in sorted(medoids.items(), key=lambda x: int(x[0]))
            ]
            pd.DataFrame(medoid_rows).to_csv(tdir / "cluster_medoids.csv", index=False)

            # Summary text
            (tdir / "cluster_summary.txt").write_text(
                "\n".join(
                    [
                        f"template_id: {template_id}",
                        f"n_structures: {len(ids)}",
                        f"linkage: {spec.linkage_method}",
                        f"atom_name: {spec.atom_name}",
                        f"cutoff: {cutoff}",
                        f"n_clusters: {n_clusters}",
                        f"silhouette: {sil}",
                        "",
                        "missing_pdbs:",
                        *([f"  - {m}" for m in missing] if missing else ["  (none)"]),
                        "",
                    ]
                )
            )

            rows_all.extend(per_rows)
            medoids_all.extend(medoid_rows)

            # Also record missing PDBs as rows in global CSV (so nothing is silently dropped)
            for vid in missing:
                rows_all.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": None,
                        "is_medoid": None,
                        "cutoff": cutoff,
                        "silhouette": sil,
                        "n_clusters": n_clusters,
                        "note": "missing_pdb",
                    }
                )

        except Exception as e:
            # record failure for this template
            for vid in variant_ids:
                rows_all.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": None,
                        "is_medoid": None,
                        "cutoff": None,
                        "silhouette": None,
                        "n_clusters": None,
                        "note": f"cluster_failed: {type(e).__name__}: {e}",
                    }
                )
            continue

    # global outputs
    out_assign = outdir / "cluster_assignments.csv"
    pd.DataFrame(rows_all).to_csv(out_assign, index=False)

    out_medoids = outdir / "cluster_medoids.csv"
    pd.DataFrame(medoids_all).to_csv(out_medoids, index=False)

    print(f"[04_cluster] wrote: {out_assign}")
    print(f"[04_cluster] wrote: {out_medoids}")
    print(f"[04_cluster] per-template outputs in: {outdir / 'clusters'}")


if __name__ == "__main__":
    main()
