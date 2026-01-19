#!/usr/bin/env python3
"""
01_embed_sequences.py

Phase 2, Step 01:
- Read variants.fasta (or another FASTA of sequences)
- Select subset of variant_ids (all / ids_file / cluster_sample_k)
- Embed sequences with ESM2 via HuggingFace transformers
- Save per-variant NPZ embeddings and an index CSV

Writes (under outdir/phase2):
- embeddings/<variant_id>.npz
- embeddings_index.csv

Usage examples:

(1) Embed all variants:
python scripts/phase2/01_embed_sequences.py \
  --outdir results/run001 \
  --variants-fasta results/run001/variants.fasta \
  --mode all \
  --model facebook/esm2_t33_650M_UR50D \
  --batch-size 2

(2) Embed first k per cluster:
python scripts/phase2/01_embed_sequences.py \
  --outdir results/run001 \
  --variants-fasta results/run001/variants.fasta \
  --mode cluster_sample_k \
  --cluster-assignments results/run001/cluster_assignments.csv \
  --k-per-cluster 5

(3) Embed IDs from file:
python scripts/phase2/01_embed_sequences.py \
  --outdir results/run001 \
  --variants-fasta results/run001/variants.fasta \
  --mode ids_file \
  --ids-file ids_to_embed.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from binpat.io.fasta import iter_fasta_records
from binpat.phase2.embed_esm2 import EmbedSpec, embed_and_save_many
from binpat.phase2.selection import SelectionSpec, select_variant_ids


# check for torch
print(f"[env] torch: {torch.__version__}")
print(f"[env] cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[env] cuda version: {torch.version.cuda}")
    print(f"[env] device: {torch.device.get_device_name(0)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 Step 01: Embed sequences with ESM2.")
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--variants-fasta", required=True, type=str)

    # Selection
    p.add_argument("--mode", type=str, default="all", choices=["all", "ids_file", "cluster_sample_k"])
    p.add_argument("--ids-file", type=str, default=None)
    p.add_argument("--cluster-assignments", type=str, default=None)
    p.add_argument("--k-per-cluster", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", action="store_true", help="Randomly sample k per cluster (default is deterministic first-k).")

    # Model / embedding
    p.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--max-length", type=int, default=None, help="Optional truncation length (token space).")

    # Output behavior
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing embedding NPZs.")

    return p.parse_args()


def _read_template_cluster_map(cluster_csv: Optional[str]) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """
    Returns variant_id -> (template_id, cluster_id)
    If no CSV provided, returns {}.
    """
    if not cluster_csv:
        return {}
    df = pd.read_csv(cluster_csv)
    out: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for _, row in df.iterrows():
        vid = str(row.get("variant_id"))
        template_id = row.get("template_id")
        cluster_id = row.get("cluster_id")
        if pd.isna(cluster_id):
            cid = None
        else:
            try:
                cid = int(cluster_id)
            except Exception:
                cid = None
        out[vid] = (None if pd.isna(template_id) else str(template_id), cid)
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    # Load sequences from FASTA in file order
    seqs_all = {rec.id: rec.seq.strip() for rec in iter_fasta_records(args.variants_fasta)}
    if not seqs_all:
        raise ValueError(f"No sequences found in {args.variants_fasta}")

    all_vids = list(seqs_all.keys())

    sel_spec = SelectionSpec(
        mode=args.mode,
        ids_file=args.ids_file,
        cluster_assignments_csv=args.cluster_assignments,
        k_per_cluster=int(args.k_per_cluster),
        seed=int(args.seed),
        sample=bool(args.sample),
    )

    selected_vids = select_variant_ids(all_vids, spec=sel_spec)
    if not selected_vids:
        raise ValueError("No sequences selected for embedding (check mode/filters).")

    selected_seqs = {vid: seqs_all[vid] for vid in selected_vids}

    tc_map = _read_template_cluster_map(args.cluster_assignments)

    spec = EmbedSpec(
        model_name=args.model,
        batch_size=int(args.batch_size),
        device=args.device,
        dtype=args.dtype,
        overwrite=bool(args.overwrite),
        max_length=(None if args.max_length is None else int(args.max_length)),
    )

    print(f"[phase2/01_embed] selected: {len(selected_seqs)} sequences")
    print(f"[phase2/01_embed] model: {spec.model_name} device={spec.device} batch={spec.batch_size} dtype={spec.dtype}")
    print(f"[phase2/01_embed] outdir: {outdir / spec.out_dirname}")

    embed_and_save_many(
        sequences=selected_seqs,
        outdir=outdir,
        spec=spec,
        template_and_cluster=tc_map if tc_map else None,
    )

    index_csv = outdir / spec.out_dirname / spec.index_filename
    print(f"[phase2/01_embed] wrote index: {index_csv}")


if __name__ == "__main__":
    main()
