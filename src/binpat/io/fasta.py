"""
fasta.py

FASTA I/O utilities for the binary-pattern + ESM2 workflow.

What's here:
- FASTA parsing.
- Validation (handled via binpat.look_up helpers).
- Helpers for "Skip + report" (ultimately implemented at call site).
- FASTA writing ("starred" records to mark sequences that were skipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from binpat.io.look_up import (
    AA20,
    FASTA_PROTEIN_ALPHABET,
    REDUCED_PATTERN_TOKENS,
    assert_valid_protein_sequence,
    reduce_to_hp_tokens,
)

PathLike = Union[str, Path]


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class FastaRecord:
    """A single FASTA record."""
    id: str
    seq: str
    description: str = ""

    @property
    def header(self) -> str:
        return f"{self.id} {self.description}".strip()


@dataclass(frozen=True)
class TokenizedRecord:
    """A FASTA record plus its reduced H/P/p/c/g token pattern."""
    id: str
    seq: str
    tokens: str
    description: str = ""

    @property
    def header(self) -> str:
        return f"{self.id} {self.description}".strip()


@dataclass(frozen=True)
class SkippedRecord:
    """Represents an input record that was skipped, with a reason."""
    id: str
    seq: str
    reason: str
    description: str = ""

    @property
    def header(self) -> str:
        return f"{self.id} {self.description}".strip()


# ----------------------------
# Core readers
# ----------------------------

def iter_fasta_records(fasta_path: PathLike) -> Iterator[FastaRecord]:
    """
    Stream FASTA records from `fasta_path`.

    Notes:
    - Accepts multi-line sequences.
    - Preserves the full header line (split into id + description).
    - Sequence is returned uppercased and with whitespace removed.
    """
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")

    current_id: Optional[str] = None
    current_desc: str = ""
    seq_chunks: List[str] = []

    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous record
                if current_id is not None:
                    seq = "".join(seq_chunks).replace(" ", "").upper()
                    yield FastaRecord(id=current_id, seq=seq, description=current_desc)
                # start new record
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header encountered in {path}")
                parts = header.split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                seq_chunks = []
            else:
                seq_chunks.append(line)

    # flush last
    if current_id is not None:
        seq = "".join(seq_chunks).replace(" ", "").upper()
        yield FastaRecord(id=current_id, seq=seq, description=current_desc)


def read_fasta_as_dict(fasta_path: PathLike) -> Dict[str, str]:
    """
    Read FASTA into a dict {id: seq}.

    If duplicate IDs are present, raises ValueError.
    """
    out: Dict[str, str] = {}
    for rec in iter_fasta_records(fasta_path):
        if rec.id in out:
            raise ValueError(f"Duplicate FASTA record id '{rec.id}' in {fasta_path}")
        out[rec.id] = rec.seq
    return out


# ----------------------------
# Validation helpers
# ----------------------------

def validate_sequence_chars(
    seq: str,
    *,
    allow_gaps: bool = False,
    allow_stop: bool = False,
) -> None:
    """
    Validate that the sequence only contains expected FASTA alphabet characters.

    This is a slightly lower-level check than `assert_valid_protein_sequence`
    because it uses FASTA_PROTEIN_ALPHABET.

    Use cases:
    - to catch weird characters early (digits, punctuation, etc.)
    - before using a strict AA20-only policy
    """
    s = seq.strip().upper()
    allowed = set(FASTA_PROTEIN_ALPHABET)
    if not allow_gaps:
        allowed.discard("-")
        allowed.discard(".")
    if not allow_stop:
        allowed.discard("*")

    bad = sorted({c for c in s if c not in allowed})
    if bad:
        raise ValueError(f"Invalid FASTA characters: {bad}")


def assert_is_aa20(seq: str) -> None:
    """
    Strictly enforce AA20-only sequences (no B/Z/X/J/U/O, no gaps, no stops).

    This is the policy used for tokenization/randomization:
    ambiguous/non-canonical codes hard-fail for that record.
    """
    s = seq.strip().upper()
    assert_valid_protein_sequence(s, allow_gaps=False, allow_stop=False)

    bad = sorted({c for c in s if c not in AA20})
    if bad:
        raise ValueError(f"Non-canonical/ambiguous residues present (AA20-only required): {bad}")


def assert_is_hp_token_pattern(
    token_seq: str,
    *,
    allow_lowercase: bool = True,
) -> None:
    """
    Validate a reduced-pattern token string.

    Allowed tokens (default) are the Option-A tokens:
        H, P, p, c, g

    Notes
    -----
    - We allow lowercase by default because convention uses lowercase for p/c/g.
    - If allow_lowercase=True, uppercase P/C/G in token patterns will be rejected (on purpose)
      to avoid confusing token patterns with AA sequences.
    """
    s = token_seq.strip().replace(" ", "")
    if not s:
        raise ValueError("Token pattern is empty.")

    allowed = set(REDUCED_PATTERN_TOKENS)

    # If someone provides lowercase, keep it; if they provide uppercase, that's fine for H/P
    # but we intentionally require p/c/g to be lowercase by convention.
    if allow_lowercase:
        # Validate character-by-character with a strict convention:
        # - H and P allowed in uppercase only
        # - p/c/g allowed in lowercase only
        for ch in s:
            if ch in {"H", "P", "p", "c", "g"}:
                continue
            raise ValueError(
                f"Invalid token '{ch}' in token pattern. Allowed tokens: H, P, p, c, g."
            )

        # Additional convention enforcement:
        # Disallow uppercase C/G/P which could be mistaken for residues.
        if any(ch in {"C", "G"} for ch in s):
            raise ValueError("Token pattern contains uppercase 'C' or 'G'. Use lowercase 'c'/'g' tokens instead.")
        # Uppercase 'P' is allowed (polar token), but uppercase 'P' as Proline is ambiguous;
        # Proline token must be 'p'. We disallow a literal residue 'P' only by policy elsewhere.
        # Here we just enforce that proline token is lowercase 'p' when intended.
        return

    # If not allowing lowercase, then only H/P are valid (pure binary)
    bad = sorted({ch for ch in s if ch not in {"H", "P"}})
    if bad:
        raise ValueError(f"Invalid tokens for pure HP pattern: {bad}. Allowed: H, P.")


# ----------------------------
# Tokenization utilities
# ----------------------------

def tokenize_records_strict(
    records: Iterable[FastaRecord],
) -> Tuple[List[TokenizedRecord], List[SkippedRecord]]:
    """
    Convert AA20 sequences -> H/P/p/c/g tokens.

    Behavior:
    - If a record contains non-AA20 characters (including ambiguous like X, B, Z, J, U, O),
      it is skipped and reported in the returned skipped list.
    """
    tokenized: List[TokenizedRecord] = []
    skipped: List[SkippedRecord] = []

    for rec in records:
        try:
            assert_is_aa20(rec.seq)
            tokens = reduce_to_hp_tokens(rec.seq)
            tokenized.append(
                TokenizedRecord(id=rec.id, seq=rec.seq, tokens=tokens, description=rec.description)
            )
        except ValueError as e:
            skipped.append(SkippedRecord(id=rec.id, seq=rec.seq, reason=str(e), description=rec.description))

    return tokenized, skipped


def read_and_tokenize_fasta_strict(
    fasta_path: PathLike,
) -> Tuple[List[TokenizedRecord], List[SkippedRecord]]:
    """Convenience wrapper: read FASTA, then tokenize with strict AA20-only policy."""
    records = list(iter_fasta_records(fasta_path))
    return tokenize_records_strict(records)


def read_hp_token_fasta_strict(
    fasta_path: PathLike,
    *,
    allow_lowercase: bool = True,
) -> Tuple[List[FastaRecord], List[SkippedRecord]]:
    """
    Read a FASTA whose sequences are *token patterns* (H/P/p/c/g).

    Returns:
        - valid token-pattern records as FastaRecord (id, seq=tokens)
        - skipped records with reasons

    Notes:
    - We return FastaRecord rather than TokenizedRecord because the "seq" is already tokens.
    - Downstream code can treat rec.seq as the token pattern.
    """
    valid: List[FastaRecord] = []
    skipped: List[SkippedRecord] = []

    for rec in iter_fasta_records(fasta_path):
        try:
            assert_is_hp_token_pattern(rec.seq, allow_lowercase=allow_lowercase)
            # Normalize: remove spaces; keep case as-is (H/P uppercase, p/c/g lowercase).
            tok = rec.seq.strip().replace(" ", "")
            valid.append(FastaRecord(id=rec.id, seq=tok, description=rec.description))
        except ValueError as e:
            skipped.append(SkippedRecord(id=rec.id, seq=rec.seq, reason=str(e), description=rec.description))

    return valid, skipped


def validate_hp_token_fasta_strict(
    fasta_path: PathLike,
    *,
    allow_lowercase: bool = True,
) -> None:
    """
    Validate that every record in a token FASTA is a valid H/P/p/c/g pattern.
    Raises ValueError if any record is invalid (fail-fast for the entire file).
    """
    valid, skipped = read_hp_token_fasta_strict(fasta_path, allow_lowercase=allow_lowercase)
    if skipped:
        msg_lines = [f"Invalid token FASTA: {len(skipped)} invalid record(s)."]
        for rec in skipped[:20]:
            msg_lines.append(f"- {rec.id}: {rec.reason}")
        if len(skipped) > 20:
            msg_lines.append(f"... plus {len(skipped) - 20} more.")
        raise ValueError("\n".join(msg_lines))
    # If no skipped, it's valid
    _ = valid


# ----------------------------
# Writers
# ----------------------------

def write_fasta(
    records: Iterable[FastaRecord],
    out_path: PathLike,
    *,
    line_width: int = 80,
) -> None:
    """Write FastaRecord objects to disk."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for rec in records:
            f.write(f">{rec.header}\n")
            seq = rec.seq.strip().upper().replace(" ", "")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


def write_token_fasta(
    token_records: Iterable[TokenizedRecord],
    out_path: PathLike,
    *,
    line_width: int = 80,
    include_original_seq_in_header: bool = False,
) -> None:
    """Write FASTA of reduced token sequences (H/P/p/c/g)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for rec in token_records:
            desc = rec.description
            if include_original_seq_in_header:
                desc = f"{desc} orig={rec.seq}".strip()
            header = f"{rec.id} {desc}".strip()
            f.write(f">{header}\n")
            tok = rec.tokens.strip()
            for i in range(0, len(tok), line_width):
                f.write(tok[i : i + line_width] + "\n")


def write_token_pattern_fasta(records: Iterable[FastaRecord], out_path: PathLike, *, line_width: int = 80) -> None:
    """Write FASTA where sequences are token patterns (preserve case)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for rec in records:
            f.write(f">{rec.header}\n")
            seq = rec.seq.strip().replace(" ", "")  # NOTE: no upper()
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


def write_variants_fasta(
    variants: Mapping[str, str],
    out_path: PathLike,
    *,
    line_width: int = 80,
    skipped: Optional[Mapping[str, str]] = None,
    star_skipped: bool = True,
) -> None:
    """
    Write a FASTA containing generated variants, with optional starred skipped entries.

    `variants`: {variant_id: variant_sequence}
    `skipped`: {variant_id: reason}
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    skipped = skipped or {}

    with path.open("w") as f:
        # Write successful variants
        for vid, vseq in variants.items():
            if vid in skipped:
                continue
            f.write(f">{vid}\n")
            seq = vseq.strip().upper().replace(" ", "")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")

        # Write skipped variants (starred)
        for vid, reason in skipped.items():
            header_id = f"*{vid}" if star_skipped else vid
            f.write(f">{header_id} skipped_reason={reason}\n")

            seq = variants.get(vid, "").strip().upper().replace(" ", "")
            if seq:
                for i in range(0, len(seq), line_width):
                    f.write(seq[i : i + line_width] + "\n")


def write_skipped_report_csv(
    skipped: Sequence[SkippedRecord],
    out_path: PathLike,
) -> None:
    """
    Write a CSV report listing skipped input FASTA records and reasons.

    Columns: sequence_id, reason, sequence
    """
    import csv

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "reason", "sequence"])
        for rec in skipped:
            w.writerow([rec.id, rec.reason, rec.seq])
