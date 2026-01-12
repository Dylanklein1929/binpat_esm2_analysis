from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure


@dataclass(frozen=True)
class AverageSpec:
    # Which atoms to use for alignment
    align_atom_names: Tuple[str, ...] = ("CA",)

    # Which atoms to average + write (must exist in all included residues)
    avg_atom_names: Tuple[str, ...] = ("N", "CA", "C")

    # Behavior
    require_same_length: bool = True
    quiet: bool = True


@dataclass(frozen=True)
class AverageResult:
    template_id: str
    cluster_id: int
    n_total: int
    n_used: int
    n_skipped: int
    reference_variant_id: str
    out_pdb_path: str
    skipped_variant_ids: Tuple[str, ...]
    note: str = ""


def _load_structure(pdb_path: Path, *, structure_id: str, quiet: bool) -> Structure:
    parser = PDBParser(QUIET=quiet)
    return parser.get_structure(structure_id, str(pdb_path))


def _iter_residues_with_atoms(struct: Structure, atom_names: Sequence[str]) -> List[Tuple]:
    """
    Returns list of (residue, atoms_list) for standard residues that contain all requested atoms.
    """
    model = next(struct.get_models())
    chain = next(model.get_chains())  # assumes single chain
    out = []
    for res in chain.get_residues():
        # skip hetero/water
        hetflag = res.id[0]
        if hetflag != " ":
            continue
        atoms = []
        ok = True
        for name in atom_names:
            if name not in res:
                ok = False
                break
            atoms.append(res[name])
        if ok:
            out.append((res, atoms))
    return out


def _extract_atom_list(struct: Structure, atom_names: Sequence[str]) -> List[Atom]:
    """
    Flattened list of atoms across residues in consistent order for the requested atom_names.
    """
    res_atoms = _iter_residues_with_atoms(struct, atom_names)
    atoms: List[Atom] = []
    for _, a_list in res_atoms:
        atoms.extend(a_list)
    return atoms


def _extract_coords(struct: Structure, atom_names: Sequence[str]) -> np.ndarray:
    atoms = _extract_atom_list(struct, atom_names)
    return np.array([a.get_coord() for a in atoms], dtype=float)


def _apply_rotran_to_coords(coords: np.ndarray, rot: np.ndarray, tran: np.ndarray) -> np.ndarray:
    # coords: (N,3); apply y = x @ rot + tran
    return coords @ rot + tran


def average_cluster_structures(
    *,
    template_id: str,
    cluster_id: int,
    member_variant_ids: List[str],
    pdb_dir: Path,
    out_pdb: Path,
    reference_variant_id: str,
    spec: AverageSpec = AverageSpec(),
) -> AverageResult:
    """
    Align all members to reference (medoid) and average avg_atom_names coordinates.
    Writes average structure PDB to out_pdb.
    """
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(member_variant_ids)
    used_coords: List[np.ndarray] = []
    skipped: List[str] = []

    ref_path = pdb_dir / f"{reference_variant_id}.pdb"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference PDB not found: {ref_path}")

    ref_struct = _load_structure(ref_path, structure_id=reference_variant_id, quiet=spec.quiet)
    ref_align_atoms = _extract_atom_list(ref_struct, spec.align_atom_names)
    ref_avg_atoms = _extract_atom_list(ref_struct, spec.avg_atom_names)

    if len(ref_align_atoms) == 0 or len(ref_avg_atoms) == 0:
        raise ValueError(f"Reference structure has no required atoms for alignment/averaging: {ref_path}")

    ref_avg_coords = np.array([a.get_coord() for a in ref_avg_atoms], dtype=float)

    for vid in member_variant_ids:
        pdb_path = pdb_dir / f"{vid}.pdb"
        if not pdb_path.exists():
            skipped.append(vid)
            continue

        try:
            s = _load_structure(pdb_path, structure_id=vid, quiet=spec.quiet)

            mov_align_atoms = _extract_atom_list(s, spec.align_atom_names)
            mov_avg_atoms = _extract_atom_list(s, spec.avg_atom_names)

            # Require same atom counts (critical for correspondence)
            if spec.require_same_length:
                if len(mov_align_atoms) != len(ref_align_atoms) or len(mov_avg_atoms) != len(ref_avg_atoms):
                    skipped.append(vid)
                    continue

            sup = Superimposer()
            sup.set_atoms(ref_align_atoms, mov_align_atoms)
            rot, tran = sup.rotran

            mov_avg_coords = np.array([a.get_coord() for a in mov_avg_atoms], dtype=float)
            mov_avg_coords_aligned = _apply_rotran_to_coords(mov_avg_coords, rot, tran)

            used_coords.append(mov_avg_coords_aligned)

        except Exception:
            skipped.append(vid)

    n_used = len(used_coords)
    n_skipped = len(skipped)

    if n_used == 0:
        raise ValueError(f"No usable structures for template={template_id} cluster={cluster_id}")

    # Include reference structure in the average? It is already aligned to itself.
    # If the reference was in member_variant_ids, it will be included naturally.
    avg_coords = np.mean(np.stack(used_coords, axis=0), axis=0)

    # Write out: clone reference structure and replace avg_atom_names coords in order
    out_struct = ref_struct.copy()
    out_avg_atoms = _extract_atom_list(out_struct, spec.avg_atom_names)

    if len(out_avg_atoms) != avg_coords.shape[0]:
        raise ValueError("Internal mismatch: out_avg_atoms and avg_coords differ in length")

    for atom, xyz in zip(out_avg_atoms, avg_coords):
        atom.set_coord(xyz)

    io = PDBIO()
    io.set_structure(out_struct)
    io.save(str(out_pdb))

    return AverageResult(
        template_id=template_id,
        cluster_id=int(cluster_id),
        n_total=n_total,
        n_used=n_used,
        n_skipped=n_skipped,
        reference_variant_id=reference_variant_id,
        out_pdb_path=str(out_pdb),
        skipped_variant_ids=tuple(skipped),
        note="",
    )
