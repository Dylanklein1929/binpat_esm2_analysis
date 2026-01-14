#!/usr/bin/env python3
"""
01_generate_library.py

Phase 1, Step 1:
- Read either natural AA FASTA OR token-pattern FASTA (H/P/p/c/g)
- (If natural) derive token patterns (AA20-only; invalid templates skipped+reported)
- Generate N hydrophobicity-constrained variants per template
- Write:
    - token_patterns.fasta (default ON)
    - variants.fasta (all variants)
    - per_template_fastas/<template_id>.fasta (default ON)
    - variants_metadata.csv
    - skipped_templates.csv

This script utilizes:
- FASTA I/O, lives in binpat.io.fasta
- generation logic, lives in binpat.phase1.library

Example:
python 01_generate_library.py --input-fasta input.fasta --config phase1.yaml --outdir outdir/

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from binpat.io.fasta import (
    FastaRecord,
    iter_fasta_records,
    read_hp_token_fasta_strict,
    write_fasta,
    write_token_pattern_fasta,
    write_skipped_report_csv,
    write_variants_fasta,
)
from binpat.phase1.library import (
    LibrarySpec,
    Phase1Outputs,
    ReplacementPools,
    Variant,
    VariantMetadata,
    generate_library_from_records,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate hydrophobicity-constrained combinatorial libraries from natural or token FASTA."
    )

    in_grp = p.add_mutually_exclusive_group(required=True)
    in_grp.add_argument("--input-fasta", type=str, help="Natural AA FASTA (AA20-only; ambiguous residues skipped).")
    in_grp.add_argument("--token-fasta", type=str, help="Token FASTA (H/P/p/c/g patterns).")

    p.add_argument("--config", type=str, required=True, help="Phase 1 YAML config (pools, num variants, seed, etc.).")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for this run.")

    # Defaults ON; allow turning off
    p.add_argument(
        "--no-token-patterns",
        action="store_true",
        help="Disable writing token_patterns.fasta (default is to write it).",
    )
    p.add_argument(
        "--no-per-template-fastas",
        action="store_true",
        help="Disable writing per-template FASTAs (default is to write them).",
    )

    return p.parse_args()

"""
    Expects a config shaped like:

    randomization:
      num_variants_per_input: 100
      rng_seed: 12345   # optional
    replacement_pools:
      hydrophobic: [A, V, I, L, M, F, W, Y]
      polar: [R, K, D, E, N, Q, S, T, H]
    """
def load_phase1_config(path: str) -> tuple[LibrarySpec, Phase1Outputs]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    rand_cfg = cfg.get("randomization", {})
    pools_cfg = cfg.get("replacement_pools", {})
    out_cfg = cfg.get("outputs", {}) or {}

    num = rand_cfg.get("num_variants_per_input", rand_cfg.get("num_variants_per_template", None))
    if num is None:
        raise ValueError("Config must set randomization.num_variants_per_input (or num_variants_per_template).")
    num = int(num)
    if num <= 0:
        raise ValueError("randomization.num_variants_per_input must be > 0.")

    seed = rand_cfg.get("rng_seed", rand_cfg.get("seed", None))
    if seed is not None:
        seed = int(seed)

    hyd = tuple(pools_cfg.get("hydrophobic", []))
    pol = tuple(pools_cfg.get("polar", []))

    spec = LibrarySpec(
        num_variants_per_template=num,
        pools=ReplacementPools(hydrophobic=hyd, polar=pol),
        rng_seed=seed,
        variant_id_prefix="var",
    )
    spec.validate()

    outputs = Phase1Outputs(
        variants_fasta_name=str(out_cfg.get("variants_fasta_name", Phase1Outputs.variants_fasta_name)),
        metadata_table_name=str(out_cfg.get("metadata_table_name", Phase1Outputs.metadata_table_name)),
    )

    return spec, outputs



def write_metadata_csv(rows: List[VariantMetadata], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write header for consistency
    if not rows:
        dummy = VariantMetadata(
            variant_id="",
            template_id="",
            variant_index=0,
            token_pattern="",
            hydrophobic_pool="",
            polar_pool="",
            recovery_to_natural=None,
        )
        fieldnames = list(asdict(dummy).keys())
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    fieldnames = list(asdict(rows[0]).keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_per_template_fastas(variants: List[Variant], outdir: Path) -> None:
    per_dir = outdir / "per_template_fastas"
    per_dir.mkdir(parents=True, exist_ok=True)

    by_template: Dict[str, List[Variant]] = {}
    for v in variants:
        by_template.setdefault(v.template_id, []).append(v)

    for tid, vs in by_template.items():
        vs_sorted = sorted(vs, key=lambda x: x.variant_index)
        recs = [FastaRecord(id=v.variant_id, seq=v.sequence) for v in vs_sorted]
        write_token_pattern_fasta(recs, per_dir / f"{tid}.fasta")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spec, outputs = load_phase1_config(args.config)

    # --- Read inputs (no redundant tokenization) ---
    if args.input_fasta:
        # Natural sequences: read records; library.py tokenizes + skips invalid AA20-only templates
        natural_records = list(iter_fasta_records(args.input_fasta))
        res = generate_library_from_records(natural_records=natural_records, spec=spec)
    else:
        # Token patterns: validate+skip invalid token patterns at read time, then generate
        token_records, skipped_in_reader = read_hp_token_fasta_strict(args.token_fasta)
        res = generate_library_from_records(token_records=token_records, spec=spec)


    # --- Write outputs ---
    # variants and metadata names
    variants_name = outputs.variants_fasta_name or "variants.fasta"
    meta_name     = outputs.metadata_table_name or "variants_metadata.csv"
    # variants and metadata paths
    variants_fasta = outdir / variants_name
    metadata_csv   = outdir / meta_name
    # others
    skipped_csv = outdir / "skipped_templates.csv"
    token_patterns_fasta = outdir / "token_patterns.fasta"

    # Skipped template report (from library result)
    write_skipped_report_csv(res.skipped_templates, skipped_csv)

    # Variants FASTA (all)
    variants_dict: Dict[str, str] = {v.variant_id: v.sequence for v in res.variants}
    write_variants_fasta(variants_dict, variants_fasta)

    # Metadata CSV
    write_metadata_csv(res.metadata, metadata_csv)

    # Token patterns FASTA (default ON)
    if not args.no_token_patterns:
        token_recs = [FastaRecord(id=t.template_id, seq=t.token_pattern, description=t.description) for t in res.templates]
        write_token_pattern_fasta(token_recs, token_patterns_fasta)

    # Per-template FASTAs (default ON)
    if not args.no_per_template_fastas:
        _write_per_template_fastas(res.variants, outdir)

    # Summary
    print(f"[01_generate_library] templates: {len(res.templates)}")
    print(f"[01_generate_library] variants:  {len(res.variants)}")
    print(f"[01_generate_library] skipped templates: {len(res.skipped_templates)}")
    print(f"[01_generate_library] wrote: {variants_fasta}")
    print(f"[01_generate_library] wrote: {metadata_csv}")
    print(f"[01_generate_library] wrote: {skipped_csv}")
    if not args.no_token_patterns:
        print(f"[01_generate_library] wrote: {token_patterns_fasta}")
    if not args.no_per_template_fastas:
        print(f"[01_generate_library] wrote per-template FASTAs to: {outdir / 'per_template_fastas'}")


if __name__ == "__main__":
    main()
