from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PCWriteResult:
    variant_id: str
    in_pdb: str
    out_pdb: str
    ok: bool
    reason: Optional[str]
    n_ca_written: int
    n_pc_available: int


def _iter_pdb_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


def _write_pdb_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _is_ca_atom_line(line: str) -> bool:
    # PDB atom name columns 13-16 (0-based 12:16)
    # Many files use " CA " for alpha carbon
    return line.startswith("ATOM") and line[12:16].strip() == "CA"


def insert_pc_into_pdb_bfactor(
    in_pdb: Path,
    out_pdb: Path,
    pc_values: np.ndarray,
) -> Tuple[int, int, Optional[str]]:
    """
    Writes pc_values (length L) into B-factor field (cols 61-66, 1-based).
    Only for CA atoms.
    Returns (n_ca_written, n_pc_available, note)
    """
    lines = _iter_pdb_lines(in_pdb)
    pc = np.asarray(pc_values).reshape(-1)
    n_pc = int(pc.shape[0])

    out_lines: List[str] = []
    i = 0
    for line in lines:
        if _is_ca_atom_line(line):
            if i < n_pc:
                # B-factor field: columns 61-66 (1-based) => 60:66 (0-based slice)
                # Keep fixed-width formatting
                new_line = line[:60] + f"{float(pc[i]):6.2f}" + line[66:]
                out_lines.append(new_line)
                i += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    note = None
    if i < n_pc:
        note = f"pc_longer_than_ca: wrote {i} CA values, had {n_pc}"
    elif i > n_pc:
        note = f"pc_shorter_than_ca: wrote {i} CA values, had {n_pc}"

    _write_pdb_lines(out_pdb, out_lines)
    return i, n_pc, note
