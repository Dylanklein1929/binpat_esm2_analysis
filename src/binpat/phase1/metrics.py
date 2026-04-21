"""
metrics.py

Phase 1 structural metrics computed from predicted PDBs.

What's here:
- Functions that compute metrics for a single PDB
- Batch helper that iterates over PDB paths and returns rows + skipped info
- Optional centroid-connectivity topology gate for structures that pass the
  hydrophobic rASA threshold
- No hard-coded directory structure; the wrapper script owns globbing / paths

Current metrics:
- helix_fraction: fraction of residues in DSSP that are helix-like (H/G/I)
- mean_hydrophobic_rasa: mean DSSP relative ASA over hydrophobic residues only
- mean_all_atom_bfactor: mean B-factor across all atoms in the structure (grand mean)
- n_res_dssp: number of residues DSSP returned
- n_hydrophobic_res: number of hydrophobic residues considered for rASA
- mean_hydrophobic_rasa_leq_threshold: raw per-structure indicator (0/1)
- topology_checked_for_rasa_success: whether centroid topology check was run
- bad_topology_among_rasa_passers: whether topology was deemed unacceptable
- rasa_success_after_topology_gate: final per-structure success (0/1) used for
  aggregate fraction_with_rasa_below_threshold

Notes:
- Requires BioPython.
- DSSP requires external mkdssp installed and on PATH.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

import logging
import warnings

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from binpat.io.look_up import DSSP_HELIX, HYDROPHOBIC_CLASSIFY

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore")


@dataclass(frozen=True)
class MetricsSpec:
    rasa_threshold: float = 0.25
    model_index: int = 0
    residues_to_skip: Set[int] = field(default_factory=set)  # PDB resseq ints (1-based)

    # Optional topology gate for rasa success
    apply_topology_check_with_rasa_gate: bool = False
    helix_ranges: Optional[List[Tuple[int, int]]] = None
    chain_id: Optional[str] = None
    abs_cosine_threshold: float = 0.6


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

    # Raw threshold flag (before topology gate)
    mean_hydrophobic_rasa_leq_threshold: Optional[int]

    # Topology gate bookkeeping
    topology_checked_for_rasa_success: Optional[int]
    bad_topology_among_rasa_passers: Optional[int]
    rasa_success_after_topology_gate: Optional[int]
    topology_abs_cosine_similarity: Optional[float]
    topology_folded_angle_degrees: Optional[float]
    topology_failure_reason: Optional[str]

    note: Optional[str] = None


@dataclass(frozen=True)
class SkippedStructure:
    variant_id: str
    pdb_path: str
    reason: str


def structure_id_from_path(pdb_path: Path) -> str:
    return pdb_path.stem


def _compute_mean_bfactor(structure, residues_to_skip: Set[int]) -> Optional[float]:
    bvals: List[float] = []
    skip = residues_to_skip or set()

    for atom in structure.get_atoms():
        residue = atom.get_parent()
        resseq = residue.get_id()[1]
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


def _get_ca_coords_for_residue_range(chain, start_resseq: int, end_resseq: int) -> np.ndarray:
    coords: List[np.ndarray] = []

    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag != " ":
            continue
        if start_resseq <= resseq <= end_resseq and res.has_id("CA"):
            coords.append(res["CA"].get_coord().astype(float))

    if not coords:
        raise ValueError(f"No CA atoms found in residue range {start_resseq}-{end_resseq}.")

    return np.asarray(coords, dtype=float)


def _centroid(coords: np.ndarray) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")
    return coords.mean(axis=0)


def _vector_between_centroids(
    chain,
    helix_a: Tuple[int, int],
    helix_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_a = _get_ca_coords_for_residue_range(chain, helix_a[0], helix_a[1])
    coords_b = _get_ca_coords_for_residue_range(chain, helix_b[0], helix_b[1])

    cen_a = _centroid(coords_a)
    cen_b = _centroid(coords_b)
    vec = cen_b - cen_a
    return cen_a, cen_b, vec


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Encountered near-zero-length vector.")
    return n


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = _safe_norm(v1)
    n2 = _safe_norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2))


def _absolute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return abs(_cosine_similarity(v1, v2))


def _angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    cosval = np.clip(_cosine_similarity(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosval)))


def _folded_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    angle = _angle_degrees(v1, v2)
    return min(angle, 180.0 - angle)


def compute_bundle_connectivity_metric(
    pdb_path: Path,
    helix_ranges: List[Tuple[int, int]],
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    if len(helix_ranges) != 4:
        raise ValueError("helix_ranges must contain exactly four helix ranges.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("bundle", str(pdb_path))
    model = next(structure.get_models())

    if chain_id is None:
        chain = next(model.get_chains())
    else:
        chain = model[chain_id]

    h1, h2, h3, h4 = helix_ranges

    c1, c2, v12 = _vector_between_centroids(chain, h1, h2)
    c3, c4, v34 = _vector_between_centroids(chain, h3, h4)

    cos_sim = _cosine_similarity(v12, v34)
    abs_cos_sim = abs(cos_sim)
    angle = _angle_degrees(v12, v34)
    folded_angle = _folded_angle_degrees(v12, v34)

    return {
        "centroid1": c1,
        "centroid2": c2,
        "centroid3": c3,
        "centroid4": c4,
        "v12": v12,
        "v34": v34,
        "cosine_similarity": cos_sim,
        "abs_cosine_similarity": abs_cos_sim,
        "angle_degrees": angle,
        "folded_angle_degrees": folded_angle,
    }


def classify_structure_for_rasa_success(
    pdb_path: Path,
    avg_hydrophobic_rasa: Optional[float],
    rasa_threshold: float,
    helix_ranges: Optional[List[Tuple[int, int]]],
    chain_id: Optional[str] = None,
    abs_cosine_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Rule:
      - If mean hydrophobic rASA is missing -> success undefined
      - If mean hydrophobic rASA > threshold:
            fail immediately, and DO NOT run topology check
      - If mean hydrophobic rASA <= threshold:
            run centroid-connectivity topology check
            - bad topology => fail
            - acceptable topology => success
    """
    result: Dict[str, Any] = {
        "passes_rasa_threshold": None,
        "topology_checked": False,
        "bad_topology": False,
        "counts_as_rasa_success": None,
        "failure_reason": None,
        "topology_metrics": None,
    }

    if avg_hydrophobic_rasa is None:
        result["failure_reason"] = "missing_mean_hydrophobic_rasa"
        return result

    result["passes_rasa_threshold"] = bool(avg_hydrophobic_rasa <= rasa_threshold)

    if avg_hydrophobic_rasa > rasa_threshold:
        result["counts_as_rasa_success"] = False
        result["failure_reason"] = "rasa_above_threshold"
        return result

    # rASA-passing structures only
    if helix_ranges is None:
        result["counts_as_rasa_success"] = True
        result["failure_reason"] = None
        return result

    result["topology_checked"] = True
    topology_metrics = compute_bundle_connectivity_metric(
        pdb_path=pdb_path,
        helix_ranges=helix_ranges,
        chain_id=chain_id,
    )
    result["topology_metrics"] = topology_metrics

    bad_topology = topology_metrics["abs_cosine_similarity"] < abs_cosine_threshold
    result["bad_topology"] = bool(bad_topology)

    if bad_topology:
        result["counts_as_rasa_success"] = False
        result["failure_reason"] = "bad_topology_among_rasa_passers"
    else:
        result["counts_as_rasa_success"] = True
        result["failure_reason"] = None

    return result


def compute_metrics_for_pdb(
    pdb_path: Path,
    *,
    spec: MetricsSpec = MetricsSpec(),
    variant_id: Optional[str] = None,
) -> StructureMetrics:
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    vid = variant_id or structure_id_from_path(pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(vid, str(pdb_path))

    models = list(structure.get_models())
    if not models:
        raise ValueError("No models found in structure.")
    if spec.model_index >= len(models):
        raise ValueError(f"Requested model_index={spec.model_index} but only {len(models)} model(s) present.")
    model = models[spec.model_index]

    dssp = DSSP(model, str(pdb_path))
    skip = spec.residues_to_skip or set()

    helix_count = 0
    hydro_rasas: List[float] = []
    included_res = 0

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
            topology_checked_for_rasa_success=None,
            bad_topology_among_rasa_passers=None,
            rasa_success_after_topology_gate=None,
            topology_abs_cosine_similarity=None,
            topology_folded_angle_degrees=None,
            topology_failure_reason="dssp_returned_zero_residues_or_all_residues_skipped",
            note="dssp_returned_zero_residues_or_all_residues_skipped",
        )

    helix_fraction = helix_count / n_res if n_res > 0 else 0.0

    if hydro_rasas:
        mean_hydro_rasa: Optional[float] = sum(hydro_rasas) / len(hydro_rasas)
        raw_leq_flag: Optional[int] = 1 if mean_hydro_rasa <= spec.rasa_threshold else 0
    else:
        mean_hydro_rasa = None
        raw_leq_flag = None

    mean_b = _compute_mean_bfactor(structure, skip)

    topology_checked = None
    bad_topology = None
    rasa_success_after_topology_gate = raw_leq_flag
    topology_abs_cos = None
    topology_folded_angle = None
    topology_failure_reason = None

    if spec.apply_topology_check_with_rasa_gate:
        topo_result = classify_structure_for_rasa_success(
            pdb_path=pdb_path,
            avg_hydrophobic_rasa=mean_hydro_rasa,
            rasa_threshold=spec.rasa_threshold,
            helix_ranges=spec.helix_ranges,
            chain_id=spec.chain_id,
            abs_cosine_threshold=spec.abs_cosine_threshold,
        )

        topology_checked = None if topo_result["topology_checked"] is None else int(bool(topo_result["topology_checked"]))
        bad_topology = None if topo_result["topology_checked"] is False else int(bool(topo_result["bad_topology"]))
        topology_failure_reason = topo_result["failure_reason"]

        if topo_result["counts_as_rasa_success"] is None:
            rasa_success_after_topology_gate = None
        else:
            rasa_success_after_topology_gate = int(bool(topo_result["counts_as_rasa_success"]))

        if topo_result["topology_metrics"] is not None:
            topology_abs_cos = float(topo_result["topology_metrics"]["abs_cosine_similarity"])
            topology_folded_angle = float(topo_result["topology_metrics"]["folded_angle_degrees"])

    return StructureMetrics(
        variant_id=vid,
        pdb_path=str(pdb_path),
        helix_fraction=float(helix_fraction),
        mean_hydrophobic_rasa=mean_hydro_rasa,
        mean_all_atom_bfactor=mean_b,
        n_res_dssp=int(n_res),
        n_helix_res=int(helix_count),
        n_hydrophobic_res=int(len(hydro_rasas)),
        mean_hydrophobic_rasa_leq_threshold=raw_leq_flag,
        topology_checked_for_rasa_success=topology_checked,
        bad_topology_among_rasa_passers=bad_topology,
        rasa_success_after_topology_gate=rasa_success_after_topology_gate,
        topology_abs_cosine_similarity=topology_abs_cos,
        topology_folded_angle_degrees=topology_folded_angle,
        topology_failure_reason=topology_failure_reason,
        note=None,
    )


def compute_metrics_batch(
    pdb_paths: Iterable[Path],
    *,
    spec: MetricsSpec = MetricsSpec(),
) -> Tuple[List[StructureMetrics], List[SkippedStructure]]:
    rows: List[StructureMetrics] = []
    skipped: List[SkippedStructure] = []

    for p in pdb_paths:
        p = Path(p)
        vid = structure_id_from_path(p)
        try:
            rows.append(compute_metrics_for_pdb(p, spec=spec, variant_id=vid))
        except Exception as e:
            skipped.append(SkippedStructure(
                variant_id=vid,
                pdb_path=str(p),
                reason=f"{type(e).__name__}: {e}",
            ))
            logger.warning("Skipping %s due to error: %s", p, e)

    return rows, skipped


def metrics_rows_to_dicts(rows: Sequence[StructureMetrics]) -> List[Dict[str, object]]:
    return [asdict(r) for r in rows]


def skipped_rows_to_dicts(rows: Sequence[SkippedStructure]) -> List[Dict[str, object]]:
    return [asdict(r) for r in rows]
