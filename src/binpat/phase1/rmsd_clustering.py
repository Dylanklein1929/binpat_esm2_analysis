from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class RMSDClusterSpec:
    atom_name: str = "CA"
    linkage_method: str = "single"   # "single", "complete", "average"
    criterion: str = "distance"

    # Cutoff search
    min_clusters: int = 2
    max_clusters: int = 20
    n_cutoffs: int = 50
    cutoff_min_quantile: float = 0.05
    cutoff_max_quantile: float = 0.95

    fixed_cutoff: Optional[float] = None


@dataclass(frozen=True)
class ClusterResult:
    ids: List[str]
    labels: List[int]
    cutoff: float
    silhouette: Optional[float]
    n_clusters: int


def _get_ca_coords_for_residue_range(
    chain,
    start_resseq: int,
    end_resseq: int,
) -> np.ndarray:
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


def filter_structs_by_centroid_connectivity_among_rasa_passers(
    pdb_paths: Dict[str, Path],
    rasa_by_id: Dict[str, float],
    rasa_threshold: float,
    helix_ranges: List[Tuple[int, int]],
    chain_id: Optional[str] = None,
    abs_cosine_threshold: float = 0.6,
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    """
    Only apply centroid-based topology filtering to structures that already pass
    the hydrophobic rASA threshold.
    """
    valid_pdb_paths: Dict[str, Path] = {}
    results_by_id: Dict[str, Dict[str, Any]] = {}

    for vid, path in pdb_paths.items():
        if vid not in rasa_by_id:
            raise KeyError(f"Missing avg_hydrophobic_rasa for structure_id '{vid}'")

        avg_rasa = float(rasa_by_id[vid])

        record: Dict[str, Any] = {
            "avg_hydrophobic_rasa": avg_rasa,
            "passes_rasa_threshold": avg_rasa <= rasa_threshold,
            "topology_checked": False,
            "bad_topology": False,
            "topology_metrics": None,
        }

        if avg_rasa <= rasa_threshold:
            record["topology_checked"] = True
            metrics = compute_bundle_connectivity_metric(
                pdb_path=path,
                helix_ranges=helix_ranges,
                chain_id=chain_id,
            )
            record["topology_metrics"] = metrics
            record["bad_topology"] = bool(
                metrics["abs_cosine_similarity"] < abs_cosine_threshold
            )

        results_by_id[vid] = record

        if not record["bad_topology"]:
            valid_pdb_paths[vid] = path

    return valid_pdb_paths, results_by_id


def _load_ca_coords(pdb_path: Path, *, structure_id: str, atom_name: str = "CA") -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, str(pdb_path))
    model = next(structure.get_models())

    coords: List[np.ndarray] = []
    for atom in model.get_atoms():
        if atom.get_name() == atom_name:
            coords.append(atom.get_coord())

    if not coords:
        raise ValueError(f"No atoms named {atom_name} found in {pdb_path}")

    return np.asarray(coords, dtype=float)


def _rmsd_from_coords(fixed: np.ndarray, moving: np.ndarray) -> float:
    if fixed.shape != moving.shape:
        raise ValueError(f"Coordinate arrays differ in shape: {fixed.shape} vs {moving.shape}")

    X = fixed
    Y = moving

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    C = Yc.T @ Xc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    Y_rot = Yc @ R

    diff = Xc - Y_rot
    return float(np.sqrt((diff * diff).sum() / X.shape[0]))


def compute_rmsd_matrix(
    pdb_paths: Dict[str, Path],
    *,
    atom_name: str = "CA",
) -> Tuple[List[str], np.ndarray]:
    ids = list(pdb_paths.keys())
    n = len(ids)
    coords: Dict[str, np.ndarray] = {}

    for sid in ids:
        coords[sid] = _load_ca_coords(pdb_paths[sid], structure_id=sid, atom_name=atom_name)

    lengths = {sid: coords[sid].shape[0] for sid in ids}
    L0 = lengths[ids[0]]
    bad = [sid for sid in ids if lengths[sid] != L0]
    if bad:
        raise ValueError(
            f"Not all structures have the same number of {atom_name} atoms. "
            f"Expected {L0}, mismatches: {[(sid, lengths[sid]) for sid in bad]}"
        )

    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = _rmsd_from_coords(coords[ids[i]], coords[ids[j]])
            mat[i, j] = rmsd
            mat[j, i] = rmsd

    return ids, mat


def compute_linkage_from_rmsd(rmsd_mat: np.ndarray, *, method: str) -> np.ndarray:
    condensed = squareform(rmsd_mat, checks=False)
    return linkage(condensed, method=method)


def cluster_labels_from_linkage(
    Z: np.ndarray,
    *,
    cutoff: float,
    criterion: str = "distance",
) -> List[int]:
    labels = fcluster(Z, t=float(cutoff), criterion=criterion).astype(int).tolist()
    return labels


def candidate_cutoffs_from_rmsd_matrix(
    rmsd_mat: np.ndarray,
    *,
    n_cutoffs: int,
    qmin: float,
    qmax: float,
) -> List[float]:
    tri = rmsd_mat[np.triu_indices(rmsd_mat.shape[0], k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return [0.0]
    lo = float(np.quantile(tri, qmin))
    hi = float(np.quantile(tri, qmax))
    if hi <= lo:
        return [lo]
    return list(np.linspace(lo, hi, n_cutoffs))


def choose_cutoff_by_silhouette(
    rmsd_mat: np.ndarray,
    Z: np.ndarray,
    *,
    criterion: str,
    cutoffs: List[float],
    min_clusters: int,
    max_clusters: int,
) -> Tuple[float, Optional[float], List[int]]:
    n = rmsd_mat.shape[0]
    best_sil = -np.inf
    best_cut = cutoffs[-1]
    best_labels = cluster_labels_from_linkage(Z, cutoff=best_cut, criterion=criterion)

    for c in cutoffs:
        labels = cluster_labels_from_linkage(Z, cutoff=float(c), criterion=criterion)
        k = len(set(labels))
        if k < min_clusters or k > min(max_clusters, n - 1):
            continue

        try:
            sil = float(silhouette_score(rmsd_mat, labels, metric="precomputed"))
        except Exception:
            continue

        if sil > best_sil:
            best_sil = sil
            best_cut = float(c)
            best_labels = labels

    if best_sil == -np.inf:
        return best_cut, None, best_labels

    return best_cut, best_sil, best_labels


def compute_cluster_medoids(
    ids: List[str],
    rmsd_mat: np.ndarray,
    labels: List[int],
) -> Dict[int, str]:
    labels_arr = np.asarray(labels, dtype=int)
    medoids: Dict[int, str] = {}

    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        if idx.size == 1:
            medoids[cid] = ids[int(idx[0])]
            continue

        sub = rmsd_mat[np.ix_(idx, idx)]
        sums = sub.sum(axis=1)
        best_local = int(np.argmin(sums))
        medoids[cid] = ids[int(idx[best_local])]

    return medoids
