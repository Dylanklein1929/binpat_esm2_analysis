from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score




@dataclass(frozen=True)
class RMSDClusterSpec:
    atom_name: str = "CA"
    linkage_method: str = "single"   # "single", "complete", "average"
    criterion: str = "distance"       # fcluster criterion

    # Cutoff search
    min_clusters: int = 2
    max_clusters: int = 20
    n_cutoffs: int = 50               # number of candidate cutoffs to test
    cutoff_min_quantile: float = 0.05
    cutoff_max_quantile: float = 0.95

    # If set, skip cutoff search and use this distance directly
    fixed_cutoff: Optional[float] = None


@dataclass(frozen=True)
class ClusterResult:
    ids: List[str]                 # structure ids in the same order as labels
    labels: List[int]              # cluster labels (1..K)
    cutoff: float                  # chosen cutoff distance
    silhouette: Optional[float]    # None if undefined
    n_clusters: int

############# test for bad/criss-cross formations (option 1): treat glycine loops as poly-line objects ##################
def _segment_segment_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Return the minimum distance between two 3D line segments p(s)=p0+s*(p1-p0),
    q(t)=q0+t*(q1-q0), with s,t in [0,1].

    Returns:
        distance, s_clamped, t_clamped
    """
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b

    # Default parameters
    s = 0.0
    t = 0.0

    if a < eps and c < eps:
        # both segments are effectively points
        return float(np.linalg.norm(p0 - q0)), 0.0, 0.0
    elif a < eps:
        # first segment is effectively a point
        t = np.clip(e / c, 0.0, 1.0) if c > eps else 0.0
        closest_p = p0
        closest_q = q0 + t * v
        return float(np.linalg.norm(closest_p - closest_q)), 0.0, float(t)
    elif c < eps:
        # second segment is effectively a point
        s = np.clip(-d / a, 0.0, 1.0) if a > eps else 0.0
        closest_p = p0 + s * u
        closest_q = q0
        return float(np.linalg.norm(closest_p - closest_q)), float(s), 0.0

    if abs(denom) > eps:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    else:
        # nearly parallel
        s = 0.0
        t = e / c if c > eps else 0.0

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # Recompute after clamping
    closest_p = p0 + s * u
    closest_q = q0 + t * v

    # One more refinement pass after clamping
    # This helps when the unconstrained solution lies outside [0,1]
    if abs(denom) > eps:
        if s in (0.0, 1.0):
            t = np.clip((b * s + e) / c, 0.0, 1.0)
            closest_p = p0 + s * u
            closest_q = q0 + t * v
        if t in (0.0, 1.0):
            s = np.clip((b * t - d) / a, 0.0, 1.0)
            closest_p = p0 + s * u
            closest_q = q0 + t * v

    dist = np.linalg.norm(closest_p - closest_q)
    return float(dist), float(s), float(t)


def _segment_angle_cosine(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Return |cos(theta)| between segment direction vectors.
    Smaller means more perpendicular; larger means more parallel.
    """
    u = p1 - p0
    v = q1 - q0
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return 1.0
    return float(abs(np.dot(u, v) / (nu * nv)))


def _extract_gly_loops_from_model(model, expected_loop_len: int = 6) -> List[List[np.ndarray]]:
    """
    Extract consecutive GLY runs of exactly expected_loop_len residues.
    Uses CA coordinates only.

    Returns:
        A list of loops, where each loop is a list of CA coordinate arrays.
    """
    loops: List[List[np.ndarray]] = []

    for chain in model:
        current_run: List[np.ndarray] = []

        for res in chain.get_residues():
            # Ignore hetero residues / waters
            hetflag = res.id[0]
            if hetflag != " ":
                if len(current_run) == expected_loop_len:
                    loops.append(current_run)
                current_run = []
                continue

            if res.get_resname() == "GLY" and res.has_id("CA"):
                current_run.append(res["CA"].get_coord().astype(float))
            else:
                if len(current_run) == expected_loop_len:
                    loops.append(current_run)
                current_run = []

        # flush end of chain
        if len(current_run) == expected_loop_len:
            loops.append(current_run)

    return loops


def _loops_criss_cross(
    loop1: List[np.ndarray],
    loop2: List[np.ndarray],
    distance_cutoff: float = 4.5,
    parallel_cosine_cutoff: float = 0.85,
    require_interior: bool = True,
) -> bool:
    """
    Decide whether two glycine loops have a 'bad criss-cross' geometry.

    Strategy:
      - Represent each loop as a polyline through CA atoms.
      - Compare every segment pair between the two loops.
      - Flag as bad if:
            * segment-segment minimum distance <= distance_cutoff
            * segments are not too parallel
            * optionally, closest points lie in the interiors of both segments

    Args:
        loop1, loop2:
            Lists of CA coordinates for the two loops
        distance_cutoff:
            Max 3D distance (Å) to consider suspicious
        parallel_cosine_cutoff:
            If |cos(theta)| is above this, segments are treated as too parallel
        require_interior:
            If True, only count close approaches where the closest points lie
            away from segment endpoints

    Returns:
        True if the loops appear to criss-cross badly, else False.
    """
    for i in range(len(loop1) - 1):
        p0 = loop1[i]
        p1 = loop1[i + 1]

        for j in range(len(loop2) - 1):
            q0 = loop2[j]
            q1 = loop2[j + 1]

            dist, s, t = _segment_segment_distance(p0, p1, q0, q1)
            if dist > distance_cutoff:
                continue

            cosang = _segment_angle_cosine(p0, p1, q0, q1)
            if cosang > parallel_cosine_cutoff:
                continue

            if require_interior:
                # Exclude cases where the closest approach happens at segment tips
                if not (0.05 < s < 0.95 and 0.05 < t < 0.95):
                    continue

            return True

    return False


def remove_structs_with_criss_crossing_loops(
    pdb_paths: Dict[str, Path],
    expected_loop_len: int = 6,
    distance_cutoff: float = 4.5,
    parallel_cosine_cutoff: float = 0.85,
    require_interior: bool = True,
) -> Dict[str, Path]:
    """
    Omit pdb structures with bad loop formations (criss-crossing glycine loops).

    Args:
        pdb_paths:
            Mapping structure_id -> pdb_path
        expected_loop_len:
            Number of consecutive glycines expected per loop
        distance_cutoff:
            Maximum segment-segment distance (Å) for suspicious proximity
        parallel_cosine_cutoff:
            Segments with |cos(theta)| above this are considered too parallel
            to count as a criss-cross
        require_interior:
            If True, closest approach must occur away from segment endpoints

    Returns:
        valid_pdb_paths:
            Dict[str, Path] of structures that pass the filter
    """
    parser = PDBParser(QUIET=True)
    valid_pdb_paths: Dict[str, Path] = {}

    for vid, path in pdb_paths.items():
        structure = parser.get_structure(vid, str(path))
        model = next(structure.get_models())

        loops = _extract_gly_loops_from_model(model, expected_loop_len=expected_loop_len)

        # If fewer than 2 loops, nothing to compare
        if len(loops) < 2:
            valid_pdb_paths[vid] = path
            continue

        bad_structure = False
        for i in range(len(loops)):
            for j in range(i + 1, len(loops)):
                if _loops_criss_cross(
                    loops[i],
                    loops[j],
                    distance_cutoff=distance_cutoff,
                    parallel_cosine_cutoff=parallel_cosine_cutoff,
                    require_interior=require_interior,
                ):
                    bad_structure = True
                    break
            if bad_structure:
                break

        if not bad_structure:
            valid_pdb_paths[vid] = path

    return valid_pdb_paths
    
###########################################################################################################################
# check for bad/criss-cross formations (option 2): compute absolute cosine similarity between helix12 and helix34 vectors #
###########################################################################################################################
def _get_ca_coords_for_residue_range(
    chain,
    start_resseq: int,
    end_resseq: int,
) -> np.ndarray:
    """
    Collect CA coordinates for residues in the inclusive PDB residue-number range
    [start_resseq, end_resseq].

    Notes:
        - Only standard residues with CA atoms are included.
        - Residue numbering is assumed to use PDB residue sequence numbers
          (res.id[1]).
        - Insertion codes are ignored.

    Args:
        chain:
            Bio.PDB Chain object
        start_resseq:
            Starting residue number (inclusive)
        end_resseq:
            Ending residue number (inclusive)

    Returns:
        coords:
            Array of shape (N, 3)

    Raises:
        ValueError if no CA atoms are found in the requested range.
    """
    coords: List[np.ndarray] = []

    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag != " ":
            continue
        if start_resseq <= resseq <= end_resseq and res.has_id("CA"):
            coords.append(res["CA"].get_coord().astype(float))

    if not coords:
        raise ValueError(
            f"No CA atoms found in residue range {start_resseq}-{end_resseq}."
        )

    return np.asarray(coords, dtype=float)


def _centroid(coords: np.ndarray) -> np.ndarray:
    """
    Compute centroid of an (N,3) coordinate array.
    """
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")
    return coords.mean(axis=0)


def _vector_between_centroids(
    chain,
    helix_a: Tuple[int, int],
    helix_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the directed vector from centroid(helix_a) to centroid(helix_b).

    Args:
        chain:
            Bio.PDB Chain object
        helix_a:
            (start_resseq, end_resseq) for helix A
        helix_b:
            (start_resseq, end_resseq) for helix B

    Returns:
        centroid_a, centroid_b, vector_b_minus_a
    """
    coords_a = _get_ca_coords_for_residue_range(chain, helix_a[0], helix_a[1])
    coords_b = _get_ca_coords_for_residue_range(chain, helix_b[0], helix_b[1])

    cen_a = _centroid(coords_a)
    cen_b = _centroid(coords_b)
    vec = cen_b - cen_a
    return cen_a, cen_b, vec


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute vector norm and guard against zero-length vectors.
    """
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Encountered near-zero-length vector.")
    return n


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Standard cosine similarity in [-1, 1].
    """
    n1 = _safe_norm(v1)
    n2 = _safe_norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2))


def _absolute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Absolute cosine similarity in [0, 1].

    Interpretation:
        1.0 -> parallel or anti-parallel
        0.0 -> orthogonal
    """
    return abs(_cosine_similarity(v1, v2))


def _angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle in degrees between two vectors, in [0, 180].
    """
    cosval = np.clip(_cosine_similarity(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosval)))


def _folded_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between two vectors folded into [0, 90].

    Parallel and anti-parallel both map near 0 degrees.
    Orthogonal maps to 90 degrees.

    Useful when anti-parallel is considered equivalent to parallel for topology.
    """
    angle = _angle_degrees(v1, v2)
    return min(angle, 180.0 - angle)


def compute_bundle_connectivity_metric(
    pdb_path: Path,
    helix_ranges: List[Tuple[int, int]],
    chain_id: Optional[str] = None,
) -> Dict[str, np.ndarray | float]:
    """
    Compute the centroid-vector alignment metric for a four-helix bundle.

    Specifically computes:
        v12 = centroid(helix1) -> centroid(helix2)
        v34 = centroid(helix3) -> centroid(helix4)

    and returns their cosine-based relationships.

    Args:
        pdb_path:
            Path to PDB file
        helix_ranges:
            List of four (start_resseq, end_resseq) tuples, in helix order:
            [helix1, helix2, helix3, helix4]
        chain_id:
            Optional chain ID. If None, uses the first chain in the first model.

    Returns:
        dict with:
            centroid1, centroid2, centroid3, centroid4 : np.ndarray shape (3,)
            v12, v34 : np.ndarray shape (3,)
            cosine_similarity : float in [-1,1]
            abs_cosine_similarity : float in [0,1]
            angle_degrees : float in [0,180]
            folded_angle_degrees : float in [0,90]
    """
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


def is_bad_criss_cross_topology_from_centroids(
    pdb_path: Path,
    helix_ranges: List[Tuple[int, int]],
    chain_id: Optional[str] = None,
    abs_cosine_threshold: float = 0.5,
) -> Tuple[bool, Dict[str, np.ndarray | float]]:
    """
    Classify a structure as bad if the helix-centroid connectivity vectors are
    insufficiently parallel.

    Args:
        pdb_path:
            Path to PDB file
        helix_ranges:
            [(h1_start, h1_end), (h2_start, h2_end), (h3_start, h3_end), (h4_start, h4_end)]
        chain_id:
            Optional chain ID
        abs_cosine_threshold:
            Structures with abs(cosine(v12, v34)) below this are flagged as bad.

            Example interpretation:
                0.9  -> very strict; requires strong parallelism
                0.7  -> moderate
                0.5  -> fairly permissive
                0.0  -> everything passes

    Returns:
        (is_bad, metrics_dict)
    """
    metrics = compute_bundle_connectivity_metric(
        pdb_path=pdb_path,
        helix_ranges=helix_ranges,
        chain_id=chain_id,
    )

    is_bad = metrics["abs_cosine_similarity"] < abs_cosine_threshold
    return bool(is_bad), metrics


def filter_structs_by_centroid_connectivity(
    pdb_paths: Dict[str, Path],
    helix_ranges: List[Tuple[int, int]],
    chain_id: Optional[str] = None,
    abs_cosine_threshold: float = 0.5,
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, np.ndarray | float]]]:
    """
    Filter out structures whose helix centroid connectivity looks like the
    criss-cross topology.

    Args:
        pdb_paths:
            Mapping structure_id -> pdb_path
        helix_ranges:
            List of four residue ranges, one per helix
        chain_id:
            Optional chain ID
        abs_cosine_threshold:
            Minimum allowed absolute cosine similarity between v12 and v34

    Returns:
        valid_pdb_paths:
            Structures that pass the filter
        metrics_by_id:
            Per-structure metrics for inspection/logging
    """
    valid_pdb_paths: Dict[str, Path] = {}
    metrics_by_id: Dict[str, Dict[str, np.ndarray | float]] = {}

    for vid, path in pdb_paths.items():
        is_bad, metrics = is_bad_criss_cross_topology_from_centroids(
            pdb_path=path,
            helix_ranges=helix_ranges,
            chain_id=chain_id,
            abs_cosine_threshold=abs_cosine_threshold,
        )
        metrics_by_id[vid] = metrics

        if not is_bad:
            valid_pdb_paths[vid] = path

    return valid_pdb_paths, metrics_by_id
##########################################################################################################################

def _load_ca_coords(pdb_path: Path, *, structure_id: str, atom_name: str = "CA") -> np.ndarray:
    """
    Load CA coordinates as an (N,3) float array.
    Assumes one model; uses first model found.
    """
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
    """
    Superpose moving onto fixed using Bio.PDB.Superimposer and return RMSD.
    """
    if fixed.shape != moving.shape:
        raise ValueError(f"Coordinate arrays differ in shape: {fixed.shape} vs {moving.shape}")

    # Using Kabsch algorithm for aligning structures; more lightweight/faster
    # than biopython's Superimposer() tool which must operate on Atom objects.
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
    """
    Compute NxN RMSD matrix (CA RMSD after optimal superposition).
    Args:
        pdb_paths: mapping structure_id -> pdb_path
    Returns:
        (ids, rmsd_matrix)
    """
    ids = list(pdb_paths.keys())
    n = len(ids)
    coords: Dict[str, np.ndarray] = {}

    for sid in ids:
        coords[sid] = _load_ca_coords(pdb_paths[sid], structure_id=sid, atom_name=atom_name)

    # Validate same length (same number of CA atoms)
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
        mat[i, i] = 0.0
        for j in range(i + 1, n):
            rmsd = _rmsd_from_coords(coords[ids[i]], coords[ids[j]])
            mat[i, j] = rmsd
            mat[j, i] = rmsd

    return ids, mat


def _candidate_cutoffs_from_matrix(
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


def cluster_from_rmsd_matrix(
    ids: List[str],
    rmsd_mat: np.ndarray,
    *,
    spec: RMSDClusterSpec = RMSDClusterSpec(),
) -> ClusterResult:
    """
    Hierarchical clustering using linkage on RMSD distances + cutoff selection by silhouette.
    """
    if rmsd_mat.shape[0] != rmsd_mat.shape[1]:
        raise ValueError("rmsd_mat must be square")
    if rmsd_mat.shape[0] != len(ids):
        raise ValueError("ids length must match rmsd_mat size")

    if len(ids) < 2:
        return ClusterResult(ids=ids, labels=[1] * len(ids), cutoff=0.0, silhouette=None, n_clusters=1)

    condensed = squareform(rmsd_mat, checks=False)
    Z = linkage(condensed, method=spec.linkage_method)

    # Choose cutoff
    if spec.fixed_cutoff is not None:
        cutoff = float(spec.fixed_cutoff)
        labels = fcluster(Z, t=cutoff, criterion=spec.criterion).astype(int).tolist()
        n_clusters = len(set(labels))
        sil = None
        if n_clusters >= 2 and n_clusters < len(ids):
            sil = float(silhouette_score(rmsd_mat, labels, metric="precomputed"))
        return ClusterResult(ids=ids, labels=labels, cutoff=cutoff, silhouette=sil, n_clusters=n_clusters)

    cutoffs = _candidate_cutoffs_from_matrix(
        rmsd_mat,
        n_cutoffs=spec.n_cutoffs,
        qmin=spec.cutoff_min_quantile,
        qmax=spec.cutoff_max_quantile,
    )

    best = (-np.inf, None, None)  # (sil, cutoff, labels)
    for c in cutoffs:
        labels = fcluster(Z, t=float(c), criterion=spec.criterion).astype(int).tolist()
        k = len(set(labels))
        if k < spec.min_clusters or k > min(spec.max_clusters, len(ids) - 1):
            continue

        try:
            sil = float(silhouette_score(rmsd_mat, labels, metric="precomputed"))
        except Exception:
            continue

        if sil > best[0]:
            best = (sil, float(c), labels)

    if best[1] is None:
        # fallback: one cluster at max cutoff
        cutoff = float(max(cutoffs))
        labels = fcluster(Z, t=cutoff, criterion=spec.criterion).astype(int).tolist()
        k = len(set(labels))
        return ClusterResult(ids=ids, labels=labels, cutoff=cutoff, silhouette=None, n_clusters=k)

    sil, cutoff, labels = best
    k = len(set(labels))
    return ClusterResult(ids=ids, labels=labels, cutoff=cutoff, silhouette=float(sil), n_clusters=k)


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


def compute_linkage_from_rmsd(rmsd_mat: np.ndarray, *, method: str) -> np.ndarray:
    """
    Correct SciPy usage: linkage expects condensed distances, not NxN square matrix.
    """
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
    """
    Returns: (best_cutoff, best_silhouette, best_labels)
    Silhouette computed with metric='precomputed' since rmsd_mat is a distance matrix.
    """
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
    """
    Medoid per cluster = structure with minimal sum of pairwise distances
    to all other members of the same cluster.
    Returns mapping: cluster_id -> medoid_variant_id
    """
    labels_arr = np.asarray(labels, dtype=int)
    medoids: Dict[int, str] = {}

    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        if idx.size == 1:
            medoids[cid] = ids[int(idx[0])]
            continue

        sub = rmsd_mat[np.ix_(idx, idx)]
        # sum distances per candidate within cluster
        sums = sub.sum(axis=1)
        best_local = int(np.argmin(sums))
        medoids[cid] = ids[int(idx[best_local])]

    return medoids
