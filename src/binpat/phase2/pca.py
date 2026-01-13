from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import IncrementalPCA


@dataclass(frozen=True)
class PCASpec:
    n_components: int = 10
    batch_residues: int = 200_000  # number of residue vectors per partial_fit batch
    dtype: str = "float32"         # cast embeddings to this dtype for PCA
    center: bool = True            # PCA always centers; keep for clarity
    max_residues_total: Optional[int] = None  # cap total residues processed (for huge runs)


@dataclass(frozen=True)
class PCAFitReport:
    n_components: int
    n_sequences: int
    n_residues_used: int
    embedding_dim: int
    explained_variance_ratio: List[float]
    singular_values: List[float]


def iter_embedding_files(embeddings_dir: Path) -> Iterator[Path]:
    for p in sorted(embeddings_dir.glob("*.npz")):
        yield p


def _load_npz_emb(path: Path, *, emb_key: str = "emb") -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if emb_key not in z:
            raise KeyError(f"{path.name}: missing key '{emb_key}'. Keys: {list(z.keys())}")
        emb = z[emb_key]
    if emb.ndim != 2:
        raise ValueError(f"{path.name}: emb must be 2D (L,D). Got shape {emb.shape}")
    return emb


def _iter_residue_batches(
    paths: Sequence[Path],
    *,
    emb_key: str,
    batch_residues: int,
    dtype: str,
    max_residues_total: Optional[int],
) -> Iterator[np.ndarray]:
    """
    Yields arrays shaped (N, D) where N ~= batch_residues (except final).
    Reads NPZs one-by-one, concatenating until batch is full.
    """
    buf: List[np.ndarray] = []
    buf_n = 0
    total = 0

    for p in paths:
        emb = _load_npz_emb(p, emb_key=emb_key).astype(dtype, copy=False)
        L, D = emb.shape

        # If global cap, truncate emb to what's left
        if max_residues_total is not None:
            remaining = max_residues_total - total
            if remaining <= 0:
                break
            if L > remaining:
                emb = emb[:remaining, :]
                L = emb.shape[0]

        buf.append(emb)
        buf_n += L
        total += L

        if buf_n >= batch_residues:
            X = np.concatenate(buf, axis=0)
            yield X
            buf = []
            buf_n = 0

    if buf_n > 0:
        yield np.concatenate(buf, axis=0)


def fit_incremental_pca(
    embeddings_dir: Path,
    *,
    spec: PCASpec,
    ids: Optional[Sequence[str]] = None,
    emb_key: str = "emb",
) -> Tuple[IncrementalPCA, PCAFitReport]:
    """
    Fits IncrementalPCA over residue embeddings stored as NPZs in embeddings_dir.
    If ids is provided, only reads <id>.npz.
    """
    if ids is None:
        paths = list(iter_embedding_files(embeddings_dir))
    else:
        paths = [(embeddings_dir / f"{vid}.npz") for vid in ids]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} embedding files. Example: {missing[0]}")

    if not paths:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    # Determine embedding dimension from first file
    first = _load_npz_emb(paths[0], emb_key=emb_key)
    D = int(first.shape[1])

    ipca = IncrementalPCA(n_components=int(spec.n_components))

    n_res_used = 0
    n_seq_used = 0

    for X in _iter_residue_batches(
        paths,
        emb_key=emb_key,
        batch_residues=int(spec.batch_residues),
        dtype=str(spec.dtype),
        max_residues_total=spec.max_residues_total,
    ):
        if X.shape[1] != D:
            raise ValueError(f"Inconsistent embedding dimension: expected {D}, got {X.shape[1]}")
        ipca.partial_fit(X)
        n_res_used += int(X.shape[0])
        # count sequences approximately: increment when we first touch each file
        # (best-effort: we’ll set n_seq_used to total number of files used at end)
    n_seq_used = len(paths)

    report = PCAFitReport(
        n_components=int(spec.n_components),
        n_sequences=int(n_seq_used),
        n_residues_used=int(n_res_used),
        embedding_dim=int(D),
        explained_variance_ratio=[float(x) for x in getattr(ipca, "explained_variance_ratio_", [])],
        singular_values=[float(x) for x in getattr(ipca, "singular_values_", [])],
    )
    return ipca, report


def write_pca_artifacts(
    out_dir: Path,
    *,
    pca: IncrementalPCA,
    report: PCAFitReport,
    model_name: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "pca_model.joblib"
    dump({"pca": pca, "model_name": model_name}, model_path)

    report_path = out_dir / "pca_report.csv"
    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n_components": report.n_components,
                "n_sequences": report.n_sequences,
                "n_residues_used": report.n_residues_used,
                "embedding_dim": report.embedding_dim,
                "explained_variance_ratio": ";".join(f"{x:.8f}" for x in report.explained_variance_ratio),
                "singular_values": ";".join(f"{x:.8f}" for x in report.singular_values),
            }
        ]
    ).to_csv(report_path, index=False)

    return {"model": model_path, "report": report_path}


def project_one(
    pca: IncrementalPCA,
    emb: np.ndarray,
    *,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Project residue embeddings to PC scores.
    Returns (L, n_components).
    """
    X = emb.astype(dtype, copy=False)
    return pca.transform(X).astype(np.float32, copy=False)


def write_scores_npz(
    out_scores_dir: Path,
    variant_id: str,
    pc_scores: np.ndarray,
) -> Path:
    out_scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_scores_dir / f"{variant_id}.npz"
    np.savez_compressed(out_path, pc=pc_scores)
    return out_path
