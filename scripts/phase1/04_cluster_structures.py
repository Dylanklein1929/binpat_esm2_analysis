#!/usr/bin/env python3
"""
04_cluster_structures.py

Phase 1, Step 04:
- Cluster predicted structures by CA RMSD (hierarchical clustering) per template_id.

Behavior:
- Uses structural_metrics_per_structure.csv from Step 03
- Applies centroid-based topology filtering ONLY among structures that pass
  the hydrophobic rASA threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
    filter_structs_by_centroid_connectivity_among_rasa_passers,
)

CENTROID_HELIX_RANGES: List[Tuple[int, int]] = [(3, 10), (20, 26), (36, 44), (54, 60)]
CENTROID_ABS_COSINE_THRESHOLD: float = 0.6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster structures by CA RMSD per template.")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--variants-metadata", required=True, type=str, help="variants_metadata.csv from Step 01")
    p.add_argument("--pdb-dir", type=str, default=None, help="Default: <outdir>/pdbs")
    p.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Default: <outdir>/structural_metrics_per_structure.csv",
    )
    p.add_argument(
        "--rasa-threshold",
        type=float,
        default=0.25,
        help="Hydrophobic rASA threshold; topology filtering is only applied among passers.",
    )

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
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (outdir / "structural_metrics_per_structure.csv")

    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {metrics_csv}. Run 03_compute_structural_metrics.py first."
        )

    metrics_df = pd.read_csv(metrics_csv)
    if "variant_id" not in metrics_df.columns or "mean_hydrophobic_rasa" not in metrics_df.columns:
        raise ValueError(
            f"Metrics CSV must contain 'variant_id' and 'mean_hydrophobic_rasa'. Found: {list(metrics_df.columns)}"
        )

    rasa_by_id: Dict[str, float] = {}
    for _, row in metrics_df.iterrows():
        vid = str(row["variant_id"])
        rasa_by_id[vid] = float(row["mean_hydrophobic_rasa"]) if pd.notna(row["mean_hydrophobic_rasa"]) else float("inf")

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

        original_pdb_paths = dict(pdb_paths)
        valid_pdb_paths, topology_results_by_id = filter_structs_by_centroid_connectivity_among_rasa_passers(
            pdb_paths=original_pdb_paths,
            rasa_by_id=rasa_by_id,
            rasa_threshold=float(args.rasa_threshold),
            helix_ranges=CENTROID_HELIX_RANGES,
            chain_id=None,
            abs_cosine_threshold=CENTROID_ABS_COSINE_THRESHOLD,
        )

        filtered_out = set(original_pdb_paths) - set(valid_pdb_paths)
        pdb_paths = valid_pdb_paths

        if len(pdb_paths) < 2:
            for vid in variant_ids:
                if vid in missing:
                    note = "missing_pdb"
                elif vid in filtered_out:
                    note = "filtered_bad_topology_among_rasa_passers"
                else:
                    note = "too_few_structures_after_filtering"

                rows_all.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": None,
                        "is_medoid": None,
                        "cutoff": None,
                        "silhouette": None,
                        "n_clusters": None,
                        "note": note,
                    }
                )
            continue

        spec = RMSDClusterSpec(
            atom_name=args.atom_name,
            linkage_method=args.linkage,
            fixed_cutoff=args.fixed_cutoff,
            n_cutoffs=int(args.n_cutoffs),
            min_clusters=int(args.min_clusters),
            max_clusters=int(args.max_clusters),
        )

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
            medoids = compute_cluster_medoids(ids, rmsd_mat, labels)

            tdir = outdir / "clusters" / str(template_id)
            tdir.mkdir(parents=True, exist_ok=True)

            if args.dendrogram:
                fig = plt.figure()
                dendrogram(Z, labels=ids, orientation="top")
                fig.tight_layout()
                fig.savefig(tdir / f"dendrogram.{args.dendrogram_format}", dpi=300)
                plt.close(fig)

            per_rows: List[dict] = []
            for vid, lab in zip(ids, labels):
                per_rows.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": int(lab),
                        "is_medoid": (vid == medoids[int(lab)]),
                        "cutoff": cutoff,
                        "silhouette": sil,
                        "n_clusters": n_clusters,
                        "note": "",
                    }
                )
            pd.DataFrame(per_rows).to_csv(tdir / "cluster_assignments.csv", index=False)

            medoid_rows = [
                {"template_id": template_id, "cluster_id": int(cid), "medoid_variant_id": mid}
                for cid, mid in sorted(medoids.items(), key=lambda x: int(x[0]))
            ]
            pd.DataFrame(medoid_rows).to_csv(tdir / "cluster_medoids.csv", index=False)

            checked_ids = [vid for vid, rec in topology_results_by_id.items() if rec["topology_checked"]]
            summary_lines = [
                f"template_id: {template_id}",
                f"n_structures: {len(ids)}",
                f"linkage: {spec.linkage_method}",
                f"atom_name: {spec.atom_name}",
                f"cutoff: {cutoff}",
                f"n_clusters: {n_clusters}",
                f"silhouette: {sil}",
                f"n_topology_checked_among_rasa_passers: {len(checked_ids)}",
                f"n_filtered_bad_topology: {len(filtered_out)}",
                "",
                "missing_pdbs:",
                *([f"  - {m}" for m in missing] if missing else ["  (none)"]),
                "",
                "filtered_bad_topology_among_rasa_passers:",
                *([f"  - {vid}" for vid in sorted(filtered_out)] if filtered_out else ["  (none)"]),
                "",
            ]
            (tdir / "cluster_summary.txt").write_text("\n".join(summary_lines))

            rows_all.extend(per_rows)
            medoids_all.extend(medoid_rows)

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

            for vid in sorted(filtered_out):
                rows_all.append(
                    {
                        "template_id": template_id,
                        "variant_id": vid,
                        "cluster_id": None,
                        "is_medoid": None,
                        "cutoff": cutoff,
                        "silhouette": sil,
                        "n_clusters": n_clusters,
                        "note": "filtered_bad_topology_among_rasa_passers",
                    }
                )

        except Exception as e:
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

    out_assign = outdir / "cluster_assignments.csv"
    pd.DataFrame(rows_all).to_csv(out_assign, index=False)

    out_medoids = outdir / "cluster_medoids.csv"
    pd.DataFrame(medoids_all).to_csv(out_medoids, index=False)

    print(f"[04_cluster] wrote: {out_assign}")
    print(f"[04_cluster] wrote: {out_medoids}")
    print(f"[04_cluster] per-template outputs in: {outdir / 'clusters'}")


if __name__ == "__main__":
    main()
