"""
look_up.py

Centralized "look-up" definitions for the binary-pattern and ESM2 mapping workflow.

What's here:
- the 20 natural amino acids, plus ambiguous/noncanonical codes that may appear in FASTA
- fixed residue classifications used to build reduced patterns (hydrophobic vs polar vs special-locked)
- reduced pattern token conventions (H/P + p/c/g for Pro/Cys/Gly)
- three/one-letter code conversions
- heptad positions and typical roles
- DSSP codes
- backbone atom name conventions
- common ESM2 model names (for config sanity checks)
- lightweight helpers for validating sequences and making reduced patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional, Tuple
from types import MappingProxyType


# ----------------------------
# Amino acids / sequence basics
# ----------------------------

# 20 natural (canonical) amino acids, single-letter codes
AA20: FrozenSet[str] = frozenset(list("ACDEFGHIKLMNPQRSTVWY"))

# Common ambiguous/extended protein alphabet symbols one might encounter in FASTA
# B = D or N, Z = E or Q, X = unknown, J = I or L, U = selenocysteine, O = pyrrolysine
AA_EXTENDED: FrozenSet[str] = frozenset(AA20 | set(list("BJZXUO")))

# Gaps and stop (seen in alignments or some FASTA exports)
GAP_CHARS: FrozenSet[str] = frozenset({"-", "."})
STOP_CHARS: FrozenSet[str] = frozenset({"*"})

# FASTA valid characters for protein sequences in your workflow
FASTA_PROTEIN_ALPHABET: FrozenSet[str] = frozenset(AA_EXTENDED | GAP_CHARS | STOP_CHARS)

# 3-letter codes (useful for PDB parsing, reporting, etc.)
AA1_TO_AA3: Mapping[str, str] = MappingProxyType(
    {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }
)
AA3_TO_AA1: Mapping[str, str] = MappingProxyType({v: k for k, v in AA1_TO_AA3.items()})


# ---------------------------------------------------------
# Fixed residue classification for reduced-pattern generation
# ---------------------------------------------------------
# IMPORTANT: These sets define how you convert AA20 -> reduced tokens.
# They are intended to be stable/scientific definitions for this project.
#
# Replacement pools used during randomization should be configurable via YAML/CLI,
# but may (by default) mirror these sets.

# Hydrophobic/core-favoring residues (Kyte–Doolittle high, common coiled-coil cores)
HYDROPHOBIC_CLASSIFY: FrozenSet[str] = frozenset({"A", "V", "I", "L", "M", "F", "W", "Y"})

# Polar/charged/solvent-favoring residues (explicitly excluding Pro/Cys/Gly below)
POLAR_CLASSIFY: FrozenSet[str] = frozenset({"R", "K", "D", "E", "N", "Q", "S", "T", "H"})

# Special-locked residues: these keep their identity through the randomization step
# (Proline/Cysteine/Glycine are preserved as P/C/G).
SPECIAL_LOCKED_AA: FrozenSet[str] = frozenset({"P", "C", "G"})

# Sanity checks (fail fast if something inconsistent is edited later)
if (HYDROPHOBIC_CLASSIFY & POLAR_CLASSIFY) or (HYDROPHOBIC_CLASSIFY & SPECIAL_LOCKED_AA) or (POLAR_CLASSIFY & SPECIAL_LOCKED_AA):
    raise RuntimeError("Residue classification sets overlap; HYDROPHOBIC_CLASSIFY, POLAR_CLASSIFY, SPECIAL_LOCKED_AA must be disjoint.")

if (HYDROPHOBIC_CLASSIFY | POLAR_CLASSIFY | SPECIAL_LOCKED_AA) != AA20:
    missing = sorted(AA20 - (HYDROPHOBIC_CLASSIFY | POLAR_CLASSIFY | SPECIAL_LOCKED_AA))
    extra = sorted((HYDROPHOBIC_CLASSIFY | POLAR_CLASSIFY | SPECIAL_LOCKED_AA) - AA20)
    raise RuntimeError(f"Residue classification does not exactly cover AA20. missing={missing}, extra={extra}")


# ------------------------------------------
# Reduced-pattern conventions
# ------------------------------------------
# Reduced string is 1:1 with sequence length:
#   H = hydrophobic (position randomized from hydrophobic replacement pool)
#   P = polar/charged (position randomized from polar replacement pool)
#   p/c/g = special-locked Pro/Cys/Gly positions (mapped back to P/C/G in output sequences)

REDUCED_PATTERN_TOKENS: FrozenSet[str] = frozenset({"H", "P", "p", "c", "g"})
LOCKED_TOKENS: FrozenSet[str] = frozenset({"p", "c", "g"})

SPECIAL_AA_TO_TOKEN: Mapping[str, str] = MappingProxyType({"P": "p", "C": "c", "G": "g"})
TOKEN_TO_SPECIAL_AA: Mapping[str, str] = MappingProxyType({v: k for k, v in SPECIAL_AA_TO_TOKEN.items()})


# ----------------------------------
# Kyte–Doolittle hydrophobicity scale
# ----------------------------------

KYTE_DOOLITTLE: Mapping[str, float] = MappingProxyType(
    {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
)
KYTE_DOOLITTLE_MIN: float = min(KYTE_DOOLITTLE.values())
KYTE_DOOLITTLE_MAX: float = max(KYTE_DOOLITTLE.values())


# -----------------------------------
# Coiled-coil heptad positions / roles
# -----------------------------------

HEPTAD: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
HEPTAD_SET: FrozenSet[str] = frozenset(HEPTAD)

HEPTAD_ROLE: Mapping[str, str] = MappingProxyType(
    {
        "a": "core",
        "d": "core",
        "e": "electrostatic",
        "g": "electrostatic",
        "b": "solvent",
        "c": "solvent",
        "f": "solvent",
    }
)

HEPTAD_CORE: FrozenSet[str] = frozenset({"a", "d"})
HEPTAD_ELECTROSTATIC: FrozenSet[str] = frozenset({"e", "g"})
HEPTAD_SOLVENT: FrozenSet[str] = frozenset({"b", "c", "f"})


# --------------------------------------
# Secondary structure conventions (DSSP)
# --------------------------------------

DSSP_HELIX: FrozenSet[str] = frozenset({"H", "G", "I"})
DSSP_BETA: FrozenSet[str] = frozenset({"E", "B"})
DSSP_COIL: FrozenSet[str] = frozenset({"T", "S", " "})

DSSP_CLASS: Mapping[str, str] = MappingProxyType(
    {**{k: "helix" for k in DSSP_HELIX}, **{k: "beta" for k in DSSP_BETA}, **{k: "coil" for k in DSSP_COIL}}
)


# ----------------------------
# Atom name conventions (PDB)
# ----------------------------

BACKBONE_ATOMS: FrozenSet[str] = frozenset({"N", "CA", "C", "O"})
BACKBONE_ATOMS_WITH_H: FrozenSet[str] = frozenset({"N", "H", "HN", "CA", "C", "O"})
COMMON_SIDECHAIN_ANCHORS: FrozenSet[str] = frozenset({"CB"})

# B-factor columns in PDB ATOM/HETATM records (1-indexed, inclusive)
PDB_BFACTOR_FIELD: Tuple[int, int] = (61, 66)


# --------------------------
# Common ESM2 model names
# --------------------------

ESM2_COMMON_MODELS: FrozenSet[str] = frozenset(
    {
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
    }
)


# ----------------------------
# Lightweight helpers
# ----------------------------

def is_valid_protein_sequence(seq: str, *, allow_gaps: bool = False, allow_stop: bool = False) -> bool:
    """Return True iff `seq` contains only characters allowed by this workflow."""
    allowed = set(AA_EXTENDED)
    if allow_gaps:
        allowed |= set(GAP_CHARS)
    if allow_stop:
        allowed |= set(STOP_CHARS)
    return all((c in allowed) for c in seq.strip().upper())


def assert_valid_protein_sequence(seq: str, *, allow_gaps: bool = False, allow_stop: bool = False) -> None:
    """Raise ValueError with a helpful message if `seq` contains unexpected characters."""
    s = seq.strip().upper()
    allowed = set(AA_EXTENDED)
    if allow_gaps:
        allowed |= set(GAP_CHARS)
    if allow_stop:
        allowed |= set(STOP_CHARS)

    bad = sorted({c for c in s if c not in allowed})
    if bad:
        raise ValueError(
            f"Invalid protein sequence characters: {bad}. "
            f"Allowed: {sorted(allowed)}"
        )


def aa_to_kyte_doolittle(aa: str, default: Optional[float] = None) -> float:
    """Return Kyte–Doolittle value for a single-letter AA.

    If aa is ambiguous (B/Z/X/J/U/O) and default is None, raises KeyError.
    """
    a = aa.strip().upper()
    if a in KYTE_DOOLITTLE:
        return float(KYTE_DOOLITTLE[a])
    if default is not None:
        return float(default)
    raise KeyError(f"No Kyte–Doolittle value for amino acid '{aa}'")


def reduce_to_hp_tokens(seq: str) -> str:
    """
    Convert an AA20 sequence into reduced tokens (same length):

        H = hydrophobic (HYDROPHOBIC_CLASSIFY)
        P = polar/charged (POLAR_CLASSIFY)
        p/c/g = special-locked residues (Pro/Cys/Gly)

    Example:
        "FSDHPCG" -> "HPPHp cg" (without spaces) -> "HPPHpcg"

    Notes:
    - Ambiguous/noncanonical codes (B/Z/X/J/U/O) raise ValueError.
    - Gaps/stops are not allowed here; validate upstream if needed.
    """
    s = seq.strip().upper()
    assert_valid_protein_sequence(s, allow_gaps=False, allow_stop=False)

    out = []
    for aa in s:
        if aa in SPECIAL_AA_TO_TOKEN:
            out.append(SPECIAL_AA_TO_TOKEN[aa])
        elif aa in HYDROPHOBIC_CLASSIFY:
            out.append("H")
        elif aa in POLAR_CLASSIFY:
            out.append("P")
        else:
            # Defensive: should be unreachable because sets exactly cover AA20
            raise ValueError(f"AA '{aa}' not classified into H/P/special.")
    return "".join(out)


@dataclass(frozen=True)
class HeptadIndexer:
    """Translate between sequence offsets and heptad positions."""
    start_pos: str = "a"

    def __post_init__(self) -> None:
        if self.start_pos not in HEPTAD_SET:
            raise ValueError(f"start_pos must be one of {HEPTAD}, got '{self.start_pos}'")

    def pos(self, i: int) -> str:
        start_idx = HEPTAD.index(self.start_pos)
        return HEPTAD[(start_idx + (i % 7)) % 7]

    def role(self, i: int) -> str:
        return HEPTAD_ROLE[self.pos(i)]
