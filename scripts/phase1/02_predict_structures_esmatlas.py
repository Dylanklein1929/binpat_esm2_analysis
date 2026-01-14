#!/usr/bin/env python3
"""
02_predict_structures_esmatlas.py

Phase 1, Step 02:
- Read variants.fasta (generated in Step 01)
- Predict structures via ESM Atlas foldSequence endpoint
- Optional subset selection: --limit, --ids-file
- Writes:
    - outdir/pdbs/*.pdb
    - outdir/prediction_report.csv
    - outdir/ids_failed.txt  (one variant_id per line)

Usage:
  python scripts/phase1/02_predict_structures_esmatlas.py \
    --variants-fasta results/run001/variants.fasta \
    --outdir results/run001 \
    --sleep-between 0.2
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Set

from binpat.io.fasta import iter_fasta_records
from binpat.io.progress import Progress
from binpat.phase1.predict_esmatlas import (
    PredictionSpec,
    predict_many,
    write_failed_ids_txt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict PDBs for variant sequences using ESM Atlas API.")
    p.add_argument("--variants-fasta", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)

    # Subsetting
    p.add_argument("--limit", type=int, default=None, help="Predict only the first N sequences (after filtering).")
    p.add_argument("--ids-file", type=str, default=None, help="Path to file with one variant_id per line to predict.")

    # Endpoint/policy
    p.add_argument("--endpoint", type=str, default=None)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument("--backoff-max", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--sleep-between", type=float, default=0.2)

    # File behavior
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PDBs instead of skipping.")
    p.add_argument("--rng-seed", type=int, default=0, help="Seed for jittered backoff randomness.")

    return p.parse_args()


def _read_ids_file(path: str) -> Set[str]:
    ids: Set[str] = set()
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.add(s)
    return ids


def _select_sequences(
    seqs: Dict[str, str],
    *,
    ids_file: Optional[str],
    limit: Optional[int],
) -> Dict[str, str]:
    items = list(seqs.items())

    if ids_file:
        keep = _read_ids_file(ids_file)
        items = [(k, v) for (k, v) in items if k in keep]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        items = items[:limit]

    return dict(items)


def write_report_csv(report_path: Path, rows) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    # Always write a header
    if not rows:
        with report_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant_id", "out_pdb_path", "status", "reason", "status_code", "attempts", "elapsed_seconds"])
        return

    fieldnames = list(asdict(rows[0]).keys())
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pdb_dir = outdir / "pdbs"
    report_csv = outdir / "prediction_report.csv"
    failed_ids_txt = outdir / "ids_failed.txt"

    # Read variants FASTA
    seqs_all = {rec.id: rec.seq.strip() for rec in iter_fasta_records(args.variants_fasta)}
    if not seqs_all:
        raise ValueError(f"No sequences found in {args.variants_fasta}")

    seqs = _select_sequences(seqs_all, ids_file=args.ids_file, limit=args.limit)
    if not seqs:
        raise ValueError("After filtering (--ids-file/--limit), no sequences remain to predict.")

    spec = PredictionSpec(
        endpoint=args.endpoint or PredictionSpec().endpoint,
        timeout_seconds=float(args.timeout),
        max_retries=int(args.max_retries),
        backoff_base_seconds=float(args.backoff_base),
        backoff_max_seconds=float(args.backoff_max),
        jitter_fraction=float(args.jitter),
        sleep_between_requests=float(args.sleep_between),
        overwrite=bool(args.overwrite),
    )

    results = predict_many(sequences=seqs, out_dir=pdb_dir, spec=spec, rng_seed=int(args.rng_seed))
    write_report_csv(report_csv, results)
    write_failed_ids_txt(results, failed_ids_txt)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")

    print(f"[02_predict_structures] total_selected: {len(results)} ok: {n_ok} skipped: {n_skip} failed: {n_fail}")
    print(f"[02_predict_structures] pdb_dir: {pdb_dir}")
    print(f"[02_predict_structures] report:  {report_csv}")
    print(f"[02_predict_structures] failed_ids: {failed_ids_txt}")


if __name__ == "__main__":
    main()
