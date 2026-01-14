"""
library.py

Phase 1: combinatorial sequence library generation for binary-pattern workflow.

Responsibilities:
- Convert natural AA sequences (AA20-only) to reduced H/P/p/c/g token patterns
- Validate user-provided H/P/p/c/g token patterns
- Generate N hydrophobicity-constrained random variants per template using configurable pools
- Produce metadata for downstream steps (FASTA writing, structure prediction, metrics)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import logging

from binpat.io.fasta import FastaRecord, SkippedRecord
from binpat.io.look_up import (
    AA20,
    POLAR_CLASSIFY,
    HYDROPHOBIC_CLASSIFY,
    REDUCED_PATTERN_TOKENS,
    assert_valid_protein_sequence,
    reduce_to_hp_tokens,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Types / dataclasses
# ----------------------------

TemplateType = Literal["natural", "token"]


@dataclass(frozen=True)
class ReplacementPools:
    """Replacement pools used during randomization (experiment/config dependent)."""
    hydrophobic: Tuple[str, ...]
    polar: Tuple[str, ...]

    def validate(self) -> None:
        h = set(self.hydrophobic)
        p = set(self.polar)

        if not h:
            raise ValueError("Hydrophobic replacement pool is empty.")
        if not p:
            raise ValueError("Polar replacement pool is empty.")

        bad_h = sorted([aa for aa in h if aa not in AA20])
        bad_p = sorted([aa for aa in p if aa not in AA20])
        if bad_h:
            raise ValueError(f"Hydrophobic pool contains non-AA20 symbols: {bad_h}")
        if bad_p:
            raise ValueError(f"Polar pool contains non-AA20 symbols: {bad_p}")

        overlap = sorted(h & p)
        if overlap:
            raise ValueError(f"Hydrophobic and polar pools overlap: {overlap}")


@dataclass(frozen=True)
class LibrarySpec:
    """Parameters defining a library-generation run."""
    num_variants_per_template: int
    pools: ReplacementPools
    rng_seed: Optional[int] = None

    # For variant IDs
    variant_id_prefix: str = "var"

    def validate(self) -> None:
        if self.num_variants_per_template <= 0:
            raise ValueError("num_variants_per_template must be > 0.")
        self.pools.validate()


@dataclass(frozen=True)
class Phase1Outputs:
    variants_fasta_name: str = "phase1_variants.fasta"
    metadata_table_name: str = "phase1_metadata.csv"


@dataclass(frozen=True)
class Template:
    """Internal normalized representation of a generation template."""
    template_id: str
    template_type: TemplateType
    token_pattern: str
    # the natural AA sequence if template_type == "natural"
    natural_sequence: Optional[str] = None
    description: str = ""


@dataclass(frozen=True)
class Variant:
    """A generated variant sequence derived from a Template."""
    variant_id: str
    template_id: str
    variant_index: int
    sequence: str


@dataclass(frozen=True)
class VariantMetadata:
    """Row-like metadata describing one generated Variant."""
    variant_id: str
    template_id: str
    variant_index: int
    token_pattern: str
    hydrophobic_pool: str  # comma-joined for provenance
    polar_pool: str        # comma-joined for provenance
    recovery_to_natural: Optional[float]  # None if template was token-only


@dataclass(frozen=True)
class LibraryResult:
    """Outputs of a library-generation run."""
    templates: List[Template]
    variants: List[Variant]
    metadata: List[VariantMetadata]
    skipped_templates: List[SkippedRecord]


# ----------------------------
# Validation helpers
# ----------------------------

def _assert_is_aa20(seq: str) -> None:
    """
    Enforce AA20-only sequences (no B/Z/X/J/U/O, no gaps, no stops).
    """
    s = seq.strip().upper()
    assert_valid_protein_sequence(s, allow_gaps=False, allow_stop=False)
    bad = sorted({c for c in s if c not in AA20})
    if bad:
        raise ValueError(f"Non-canonical/ambiguous residues present (AA20-only required): {bad}")


def _assert_is_hp_token_pattern(token_seq: str) -> None:
    """
    Validate H/P/p/c/g token pattern.

    Conventions:
      - H and P must be uppercase
      - p/c/g must be lowercase
    """
    s = token_seq.strip().replace(" ", "")
    if not s:
        raise ValueError("Token pattern is empty.")

    for ch in s:
        if ch in {"H", "P", "p", "c", "g"}:
            continue
        raise ValueError(f"Invalid token '{ch}' in token pattern. Allowed tokens: H, P, p, c, g.")

    # guard against confusing residue letters with tokens
    if any(ch in {"C", "G"} for ch in s):
        raise ValueError("Token pattern contains uppercase 'C' or 'G'. Use lowercase 'c'/'g' tokens instead.")


def _sequence_recovery(a: str, b: str) -> float:
    """Fraction of identical positions."""
    if len(a) != len(b):
        raise ValueError("Sequences must be the same length to compute recovery.")
    return sum(x == y for x, y in zip(a, b)) / len(a)


# ----------------------------
# Template normalization
# ----------------------------

def templates_from_natural_records(
    records: Iterable[FastaRecord],
) -> Tuple[List[Template], List[SkippedRecord]]:
    """
    Convert natural AA FASTA records into Templates (tokenized).
    Invalid AA20-only sequences are skipped and reported.
    """
    templates: List[Template] = []
    skipped: List[SkippedRecord] = []

    for rec in records:
        try:
            _assert_is_aa20(rec.seq)
            tokens = reduce_to_hp_tokens(rec.seq)  # yields H/P/p/c/g
            templates.append(
                Template(
                    template_id=rec.id,
                    template_type="natural",
                    token_pattern=tokens,
                    natural_sequence=rec.seq.strip().upper(),
                    description=rec.description,
                )
            )
        except ValueError as e:
            skipped.append(SkippedRecord(id=rec.id, seq=rec.seq, reason=str(e), description=rec.description))

    return templates, skipped


def templates_from_token_records(
    records: Iterable[FastaRecord],
) -> Tuple[List[Template], List[SkippedRecord]]:
    """
    Convert token-pattern FASTA records (H/P/p/c/g) into Templates.
    Invalid token patterns are skipped and reported.
    """
    templates: List[Template] = []
    skipped: List[SkippedRecord] = []

    for rec in records:
        try:
            tok = rec.seq.strip().replace(" ", "")
            _assert_is_hp_token_pattern(tok)
            templates.append(
                Template(
                    template_id=rec.id,
                    template_type="token",
                    token_pattern=tok,
                    natural_sequence=None,
                    description=rec.description,
                )
            )
        except ValueError as e:
            skipped.append(SkippedRecord(id=rec.id, seq=rec.seq, reason=str(e), description=rec.description))

    return templates, skipped


# ----------------------------
# Variant generation
# ----------------------------

def _instantiate_from_tokens(
    token_pattern: str,
    pools: ReplacementPools,
    rng: random.Random,
) -> str:
    """
    Instantiate one concrete AA20 sequence from an H/P/p/c/g token pattern.
    p/c/g map deterministically to P/C/G.
    """
    out: List[str] = []
    for tok in token_pattern:
        if tok == "H":
            out.append(rng.choice(pools.hydrophobic))
        elif tok == "P":
            out.append(rng.choice(pools.polar))
        elif tok == "p":
            out.append("P")
        elif tok == "c":
            out.append("C")
        elif tok == "g":
            out.append("G")
        else:
            # token patterns are validated upstream
            raise ValueError(f"Unknown token '{tok}'")
    return "".join(out)


def generate_library(
    templates: Sequence[Template],
    spec: LibrarySpec,
) -> LibraryResult:
    """
    Generate a library of variants from already-normalized Templates.

    Returns LibraryResult with variants + per-variant metadata.
    """
    spec.validate()
    rng = random.Random(spec.rng_seed)

    variants: List[Variant] = []
    metadata: List[VariantMetadata] = []

    hyd_pool_str = ",".join(spec.pools.hydrophobic)
    pol_pool_str = ",".join(spec.pools.polar)

    for t in templates:
        for i in range(spec.num_variants_per_template):
            variant_id = f"{t.template_id}_{spec.variant_id_prefix}{i:04d}"
            seq = _instantiate_from_tokens(t.token_pattern, spec.pools, rng)

            variants.append(
                Variant(
                    variant_id=variant_id,
                    template_id=t.template_id,
                    variant_index=i,
                    sequence=seq,
                )
            )

            recov: Optional[float] = None
            if t.template_type == "natural" and t.natural_sequence is not None:
                recov = _sequence_recovery(t.natural_sequence, seq)

            metadata.append(
                VariantMetadata(
                    variant_id=variant_id,
                    template_id=t.template_id,
                    variant_index=i,
                    token_pattern=t.token_pattern,
                    hydrophobic_pool=hyd_pool_str,
                    polar_pool=pol_pool_str,
                    recovery_to_natural=recov,
                )
            )

    return LibraryResult(
        templates=list(templates),
        variants=variants,
        metadata=metadata,
        skipped_templates=[],
    )


def generate_library_from_records(
    *,
    natural_records: Optional[Iterable[FastaRecord]] = None,
    token_records: Optional[Iterable[FastaRecord]] = None,
    spec: LibrarySpec,
) -> LibraryResult:
    """
    One-stop convenience function:

    - If natural_records is provided: tokenize to Templates (AA20-only) and generate variants.
    - If token_records is provided: validate token patterns and generate variants.
    - Exactly one of natural_records or token_records must be provided.

    Returns LibraryResult including skipped_templates from template normalization.
    """
    if (natural_records is None) == (token_records is None):
        raise ValueError("Provide exactly one of natural_records or token_records.")

    if natural_records is not None:
        templates, skipped = templates_from_natural_records(natural_records)
    else:
        assert token_records is not None
        templates, skipped = templates_from_token_records(token_records)

    # If there are no valid templates, return early
    if not templates:
        return LibraryResult(templates=[], variants=[], metadata=[], skipped_templates=skipped)

    res = generate_library(templates, spec)
    # Attach skipped templates to result
    return LibraryResult(
        templates=res.templates,
        variants=res.variants,
        metadata=res.metadata,
        skipped_templates=skipped,
    )
