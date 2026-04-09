"""
metrics.py

Phase 1 structural metrics computed from predicted PDBs.

What's here:
- Functions that compute metrics for a single PDB
- Batch helper that iterates over PDB paths and returns rows + skipped info
- No hard-coded directory structure; the wrapper script owns globbing / paths

Current metrics:
- helix_fraction: fraction of residues in DSSP that are helix-like (H/G/I)
- mean_hydrophobic_rasa: mean DSSP relative ASA over hydrophobic residues only
- mean_all_atom_bfactor: mean B-factor across all atoms in the structure (grand mean)
- n_res_dssp: number of residues DSSP returned
- n_hydrophobic_res: number of hydrophobic residues considered for rASA
- frac_hydrophobic_rasa_leq_threshold: indicator per-structure (0/1) for mean_hydrophobic_rasa <= threshold
  (Aggregate across many structures to reproduce "fraction low rASA")

Notes:
- Requires BioPython.
- DSSP requires external mkdssp installed and on PATH.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set

import logging
import warnings

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from binpat.io.look_up import DSSP_HELIX, HYDROPHOBIC_CLASSIFY

logger = logging.getLogger(__name__)

# Suppress common structure warnings
warnings.simplefilter("ignore")


@dataclass(frozen=True)
class MetricsSpec:
    rasa_threshold: float = 0.25
    model_index: int = 0
    residues_to_skip: Set[int] = field(default_factory=set)  # store as PDB resseq ints (1-based)


@dataclass(frozen=True)
class StructureMetrics:
    """
    One row of metrics for one structure file.
    """
    variant_id: str
    pdb_path: str

    helix_fraction: float
    mean_hydrophobic_rasa: Optional[float]
    mean_all_atom_bfactor: Optional[float]

    n_res_dssp: int
    n_helix_res: int
    n_hydrophobic_res: int

    # Convenience flag: 1 if mean_hydrophobic_rasa <= threshold, else 0 (None -> None)
    mean_hydrophobic_rasa_leq_threshold: Optional[int]

    # Optional: keep a short note, typically None when ok
    note: Optional[str] = None


@dataclass(frozen=True)
class SkippedStructure:
    variant_id: str
    pdb_path: str
    reason: str


def structure_id_from_path(pdb_path: Path) -> str:
    """
    Default mapping from path to variant_id: file stem.
    e.g., pdbs/seq1_var0000.pdb -> seq1_var0000
    """
    return pdb_path.stem


def _compute_mean_bfactor(structure, residues_to_skip: Set[int]) -> Optional[float]:
    bvals: List[float] = []
    skip = residues_to_skip or set()

    for atom in structure.get_atoms():
        residue = atom.get_parent()
        resseq = residue.get_id()[1]  # PDB residue number
        if resseq in skip:
            continue
        try:
            bvals.append(float(atom.get_bfactor()))
        except Exception:
            continue

    if not bvals:
        return None
    return sum(bvals) / len(bvals)

def get_residues_to_skip_from_file(path: Path) -> Set[int]:
    """
    File format: comma-separated residue indices (1-based), e.g.:
      1,2,3,10
      25,26
    """
    out: Set[int] = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split(","):
                tok = tok.strip()
                if tok:
                    out.add(int(tok))
    return out


def compute_metrics_for_pdb(
    pdb_path: Path,
    *,
    spec: MetricsSpec = MetricsSpec(),
    variant_id: Optional[str] = None,
) -> StructureMetrics:
    """
    Compute metrics for a single PDB path.

    Raises:
        Exception if parsing/DSSP fails. (Batch runner catches and records skip.)
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    vid = variant_id or structure_id_from_path(pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(vid, str(pdb_path))

    # Choose model
    models = list(structure.get_models())
    if not models:
        raise ValueError("No models found in structure.")
    if spec.model_index >= len(models):
        raise ValueError(f"Requested model_index={spec.model_index} but only {len(models)} model(s) present.")
    model = models[spec.model_index]

    # DSSP: relies on mkdssp availability
    dssp = DSSP(model, str(pdb_path))

    skip = spec.residues_to_skip or set()

    helix_count = 0
    hydro_rasas: List[float] = []
    included_res = 0

    # Biopython DSSP tuple indices:
    # [1] aa, [2] ss, [3] rasa
    for key in dssp.keys():
        resseq = dssp[key][0]
        aa = dssp[key][1]
        ss = dssp[key][2]
        rasa = dssp[key][3]

        if resseq in skip:
            continue

        included_res += 1

        if ss in DSSP_HELIX:
            helix_count += 1

        if aa in HYDROPHOBIC_CLASSIFY:
            try:
                hydro_rasas.append(float(rasa))
            except Exception:
                pass

    n_res = included_res

    if n_res == 0:
        # Rare, but handle explicitly
        return StructureMetrics(
            variant_id=vid,
            pdb_path=str(pdb_path),
            helix_fraction=0.0,
            mean_hydrophobic_rasa=None,
            mean_all_atom_bfactor=_compute_mean_bfactor(structure, residues_to_skip=skip),
            n_res_dssp=0,
            n_helix_res=0,
            n_hydrophobic_res=0,
            mean_hydrophobic_rasa_leq_threshold=None,
            note="dssp_returned_zero_residues_or_all_residues_skipped",
        )

    helix_fraction = helix_count / n_res if n_res > 0 else 0.0

    mean_hydro_rasa: Optional[float]
    leq_flag: Optional[int]
    if hydro_rasas:
        mean_hydro_rasa = sum(hydro_rasas) / len(hydro_rasas)
        leq_flag = 1 if mean_hydro_rasa <= spec.rasa_threshold else 0
    else:
        mean_hydro_rasa = None
        leq_flag = None

    mean_b = _compute_mean_bfactor(structure, skip)

    return StructureMetrics(
        variant_id=vid,
        pdb_path=str(pdb_path),
        helix_fraction=float(helix_fraction),
        mean_hydrophobic_rasa=mean_hydro_rasa,
        mean_all_atom_bfactor=mean_b,
        n_res_dssp=int(n_res),
        n_helix_res=int(helix_count),
        n_hydrophobic_res=int(len(hydro_rasas)),
        mean_hydrophobic_rasa_leq_threshold=leq_flag,
        note=None,
    )


def compute_metrics_batch(
    pdb_paths: Iterable[Path],
    *,
    spec: MetricsSpec = MetricsSpec(),
) -> Tuple[List[StructureMetrics], List[SkippedStructure]]:
    """
    Compute metrics for many PDBs.

    Returns:
        (metrics_rows, skipped_rows)
    """
    rows: List[StructureMetrics] = []
    skipped: List[SkippedStructure] = []

    for p in pdb_paths:
        p = Path(p)
        vid = structure_id_from_path(p)
        fasta_vid = (
            vid.replace("-", ",")
               .replace("_helices_", "|helices|")
               .replace("_loops_", "|loops|")
               .replace("_patterns_", "|patterns|")
        )
        try:
            rows.append(compute_metrics_for_pdb(p, spec=spec, variant_id=fasta_vid))
        except Exception as e:
            skipped.append(SkippedStructure(
                variant_id=fasta_vid,
                pdb_path=str(p),
                reason=f"{type(e).__name__}: {e}",
            ))
            logger.warning("Skipping %s due to error: %s", p, e)

    return rows, skipped


def metrics_rows_to_dicts(rows: Sequence[StructureMetrics]) -> List[Dict[str, object]]:
    return [asdict(r) for r in rows]


def skipped_rows_to_dicts(rows: Sequence[SkippedStructure]) -> List[Dict[str, object]]:
    return [asdict(r) for r in rows]
