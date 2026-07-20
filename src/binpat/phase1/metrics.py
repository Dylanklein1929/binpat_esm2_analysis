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

import re

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

    # Optional topology gate for rASA success
    apply_topology_check_with_rasa_gate: bool = False

    # By default, derive the four helix ranges from template metadata embedded
    # in variant_id, e.g.:
    #   binpat_len_100|helices|21,20,20,21|loops|6,6,6|...
    #
    # Set derive_helix_ranges_from_variant_id=False to use helix_ranges as a
    # manual override instead.
    derive_helix_ranges_from_variant_id: bool = True
    helix_ranges: Optional[List[Tuple[int, int]]] = None

    # Number of central residues from each encoded helix to use for its
    # centroid. Set to None to use each full encoded helix.
    topology_helix_window_size: Optional[int] = 8

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

    # Audit fields showing exactly which one-based inclusive ranges were used.
    topology_helix_ranges: Optional[str]
    topology_helix_range_source: Optional[str]

    note: Optional[str] = None


@dataclass(frozen=True)
class SkippedStructure:
    variant_id: str
    pdb_path: str
    reason: str


def structure_id_from_path(pdb_path: Path) -> str:
    return pdb_path.stem


def _parse_template_lengths_from_variant_id(
    variant_id: str,
) -> Tuple[List[int], List[int]]:
    """
    Parse four helix lengths and three loop lengths from a variant/template ID.

    Expected metadata anywhere in variant_id:
        |helices|H1,H2,H3,H4|loops|L1,L2,L3|

    Example:
        binpat_len_100|helices|21,20,20,21|loops|6,6,6|pattern|...

    Returns:
        ([H1, H2, H3, H4], [L1, L2, L3])
    """
    helix_match = re.search(
        r"(?:^|\|)helices\|([0-9]+(?:,[0-9]+){3})(?:\||$)",
        variant_id,
    )
    loop_match = re.search(
        r"(?:^|\|)loops\|([0-9]+(?:,[0-9]+){2})(?:\||$)",
        variant_id,
    )

    if helix_match is None or loop_match is None:
        raise ValueError(
            "Could not parse template helix/loop metadata from variant_id: "
            f"{variant_id!r}. Expected fields like "
            "'|helices|21,20,20,21|loops|6,6,6|'."
        )

    helix_lengths = [int(x) for x in helix_match.group(1).split(",")]
    loop_lengths = [int(x) for x in loop_match.group(1).split(",")]

    if len(helix_lengths) != 4:
        raise ValueError(
            f"Expected four helix lengths, got {helix_lengths} in {variant_id!r}."
        )
    if len(loop_lengths) != 3:
        raise ValueError(
            f"Expected three loop lengths, got {loop_lengths} in {variant_id!r}."
        )
    if any(x <= 0 for x in helix_lengths):
        raise ValueError(f"Helix lengths must be positive: {helix_lengths}.")
    if any(x < 0 for x in loop_lengths):
        raise ValueError(f"Loop lengths cannot be negative: {loop_lengths}.")

    encoded_length = sum(helix_lengths) + sum(loop_lengths)

    length_match = re.search(r"(?:^|\|)binpat_len_([0-9]+)(?:\||$)", variant_id)
    if length_match is None:
        # The current IDs begin with binpat_len_N without a leading pipe.
        length_match = re.search(r"\bbinpat_len_([0-9]+)(?:\||$)", variant_id)

    if length_match is not None:
        stated_length = int(length_match.group(1))
        if stated_length != encoded_length:
            raise ValueError(
                "Template metadata is internally inconsistent for "
                f"{variant_id!r}: binpat_len_{stated_length}, but helix + loop "
                f"lengths sum to {encoded_length}."
            )

    return helix_lengths, loop_lengths


def _full_helix_ranges_from_lengths(
    helix_lengths: Sequence[int],
    loop_lengths: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Convert four helix lengths and three intervening loop lengths into
    one-based, inclusive sequence/PDB residue ranges.
    """
    if len(helix_lengths) != 4 or len(loop_lengths) != 3:
        raise ValueError(
            "Expected four helix lengths and three loop lengths, got "
            f"{list(helix_lengths)} and {list(loop_lengths)}."
        )

    ranges: List[Tuple[int, int]] = []
    start = 1

    for i, helix_length in enumerate(helix_lengths):
        end = start + int(helix_length) - 1
        ranges.append((start, end))

        if i < len(loop_lengths):
            start = end + int(loop_lengths[i]) + 1

    return ranges


def _central_residue_window(
    residue_range: Tuple[int, int],
    width: Optional[int],
) -> Tuple[int, int]:
    """
    Return a centered one-based inclusive subrange.

    width=None uses the full range. If width exceeds the helix length, the
    full helix is used.
    """
    start, end = residue_range
    length = end - start + 1

    if length <= 0:
        raise ValueError(f"Invalid residue range: {residue_range}.")
    if width is None:
        return residue_range
    if width <= 0:
        raise ValueError("topology_helix_window_size must be positive or None.")

    selected_width = min(int(width), length)
    selected_start = start + (length - selected_width) // 2
    selected_end = selected_start + selected_width - 1
    return selected_start, selected_end


def derive_topology_helix_ranges(
    variant_id: str,
    central_window_size: Optional[int] = 8,
) -> List[Tuple[int, int]]:
    """
    Derive the four topology centroid ranges from the template metadata in
    variant_id.

    The returned ranges are one-based and inclusive, matching the PDB residue
    numbers used by _get_ca_coords_for_residue_range().
    """
    helix_lengths, loop_lengths = _parse_template_lengths_from_variant_id(
        variant_id
    )
    full_ranges = _full_helix_ranges_from_lengths(
        helix_lengths,
        loop_lengths,
    )
    return [
        _central_residue_window(r, central_window_size)
        for r in full_ranges
    ]


def _format_helix_ranges(
    helix_ranges: Optional[Sequence[Tuple[int, int]]],
) -> Optional[str]:
    if helix_ranges is None:
        return None
    return ";".join(f"{start}-{end}" for start, end in helix_ranges)


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
            topology_helix_ranges=None,
            topology_helix_range_source=None,
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
    topology_helix_ranges = None
    topology_helix_range_source = None

    if spec.apply_topology_check_with_rasa_gate:
        # Preserve the intended gate order: derive and use topology ranges only
        # for structures that first pass the hydrophobic-rASA threshold.
        effective_helix_ranges = spec.helix_ranges

        if mean_hydro_rasa is not None and mean_hydro_rasa <= spec.rasa_threshold:
            if spec.derive_helix_ranges_from_variant_id:
                effective_helix_ranges = derive_topology_helix_ranges(
                    variant_id=vid,
                    central_window_size=spec.topology_helix_window_size,
                )
                topology_helix_range_source = "template_variant_id"
            elif effective_helix_ranges is not None:
                topology_helix_range_source = "manual_metrics_spec"

            topology_helix_ranges = _format_helix_ranges(
                effective_helix_ranges
            )

        topo_result = classify_structure_for_rasa_success(
            pdb_path=pdb_path,
            avg_hydrophobic_rasa=mean_hydro_rasa,
            rasa_threshold=spec.rasa_threshold,
            helix_ranges=effective_helix_ranges,
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
        topology_helix_ranges=topology_helix_ranges,
        topology_helix_range_source=topology_helix_range_source,
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
