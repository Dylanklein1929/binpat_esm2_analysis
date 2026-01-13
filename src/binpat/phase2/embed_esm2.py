"""
embed_esm2.py

Phase 2 core: embed sequences with an ESM2 model and save per-variant embeddings.

Output layout (under outdir/phase2):
- embeddings/<variant_id>.npz
- embeddings_index.csv  (append-safe report; restartable)

NPZ content:
- emb: (L, D) float16/float32 per-residue embeddings (final hidden layer by default)
- seq: stored as a string array
- model_name: string
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel


@dataclass(frozen=True)
class EmbedSpec:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    batch_size: int = 1
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    dtype: str = "float16"  # "float16" | "float32"
    overwrite: bool = False

    # Output control
    out_dirname: str = "phase2"
    embeddings_subdir: str = "embeddings"
    index_filename: str = "embeddings_index.csv"

    # Tokenization
    max_length: Optional[int] = None  # None = no truncation


@dataclass(frozen=True)
class EmbedResult:
    variant_id: str
    template_id: Optional[str]
    cluster_id: Optional[int]
    sequence_length: int
    model_name: str
    device: str
    embedding_path: str
    status: str  # ok | skipped | failed
    error: str
    elapsed_seconds: float


def _resolve_device(device: str) -> str:
    d = (device or "auto").lower().strip()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in ("cpu", "cuda"):
        if d == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return d
    raise ValueError(f"Unknown device: {device!r}")


def _resolve_dtype(dtype: str) -> np.dtype:
    d = (dtype or "").lower().strip()
    if d == "float16":
        return np.float16
    if d == "float32":
        return np.float32
    raise ValueError(f"Unknown dtype: {dtype!r}")


def _npz_path(base_outdir: Path, spec: EmbedSpec, variant_id: str) -> Path:
    return base_outdir / spec.out_dirname / spec.embeddings_subdir / f"{variant_id}.npz"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_existing_index(index_csv: Path) -> Dict[str, str]:
    """
    Returns: variant_id -> status for entries already present in the index.
    Used to support resume behavior in addition to file existence.
    """
    if not index_csv.exists():
        return {}
    out: Dict[str, str] = {}
    with index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get("variant_id") or "").strip()
            status = (row.get("status") or "").strip()
            if vid:
                out[vid] = status
    return out


def _append_results(index_csv: Path, rows: List[EmbedResult]) -> None:
    _ensure_parent(index_csv)

    file_exists = index_csv.exists()
    fieldnames = list(asdict(rows[0]).keys()) if rows else []

    with index_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _extract_residue_embeddings(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """
    hidden: (B, T, D)
    attention_mask: (B, T) with 1 for real tokens incl special tokens, 0 for pad

    Returns list length B:
      each is (L, D) embeddings without special tokens and without padding.
    """
    B = hidden.size(0)
    out: List[torch.Tensor] = []
    for i in range(B):
        mask_i = attention_mask[i].bool()  # includes BOS/EOS, excludes pad
        toks = hidden[i][mask_i]           # (T_real, D)

        # ESM2 tokenizer uses BOS/EOS -> drop first and last token
        if toks.size(0) >= 2:
            toks = toks[1:-1]
        else:
            toks = toks[:0]

        out.append(toks)
    return out


def embed_and_save_many(
    *,
    sequences: Dict[str, str],               # variant_id -> sequence
    outdir: Path,
    spec: EmbedSpec,
    template_and_cluster: Optional[Dict[str, Tuple[Optional[str], Optional[int]]]] = None,
) -> List[EmbedResult]:
    """
    Embeds sequences and writes <outdir>/phase2/embeddings/<variant_id>.npz per variant.
    Writes/updates <outdir>/phase2/embeddings_index.csv (append-safe).
    """
    outdir = Path(outdir)
    base_phase2 = outdir / spec.out_dirname
    emb_dir = base_phase2 / spec.embeddings_subdir
    index_csv = base_phase2 / spec.index_filename
    emb_dir.mkdir(parents=True, exist_ok=True)

    existing_status = _load_existing_index(index_csv)
    device = _resolve_device(spec.device)
    np_dtype = _resolve_dtype(spec.dtype)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
    model = EsmModel.from_pretrained(spec.model_name)
    model.eval()
    model.to(device)

    # batching
    vids = list(sequences.keys())
    results: List[EmbedResult] = []

    def chunk(lst: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch_vids in chunk(vids, max(1, int(spec.batch_size))):
        # Pre-skip if all exist and overwrite False
        batch_to_run: List[str] = []
        for vid in batch_vids:
            npz_path = _npz_path(outdir, spec, vid)
            if (not spec.overwrite) and npz_path.exists():
                # Still report skipped
                t_id, c_id = (None, None)
                if template_and_cluster and vid in template_and_cluster:
                    t_id, c_id = template_and_cluster[vid]
                results.append(
                    EmbedResult(
                        variant_id=vid,
                        template_id=t_id,
                        cluster_id=c_id,
                        sequence_length=len(sequences[vid]),
                        model_name=spec.model_name,
                        device=device,
                        embedding_path=str(npz_path.relative_to(outdir)),
                        status="skipped",
                        error="exists",
                        elapsed_seconds=0.0,
                    )
                )
            else:
                batch_to_run.append(vid)

        if not batch_to_run:
            # periodically flush
            if results:
                _append_results(index_csv, results)
                results = []
            continue

        seqs = [sequences[vid] for vid in batch_to_run]
        start = time.time()

        try:
            toks = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=(spec.max_length is not None),
                max_length=spec.max_length,
                add_special_tokens=True,
            )
            toks = {k: v.to(device) for k, v in toks.items()}

            with torch.no_grad():
                out = model(**toks)
            hidden = out.last_hidden_state  # (B, T, D)

            per_res = _extract_residue_embeddings(hidden, toks["attention_mask"])

            # Save each
            for vid, seq, emb_t in zip(batch_to_run, seqs, per_res):
                npz_path = _npz_path(outdir, spec, vid)
                _ensure_parent(npz_path)

                # Sanity: should match original length unless truncated
                L = emb_t.size(0)
                if spec.max_length is None and L != len(seq):
                    raise ValueError(
                        f"Length mismatch for {vid}: embeddings {L} vs seq {len(seq)}"
                    )

                emb_np = emb_t.detach().cpu().numpy().astype(np_dtype, copy=False)

                np.savez_compressed(
                    npz_path,
                    emb=emb_np,
                    seq=np.array([seq]),
                    model_name=np.array([spec.model_name]),
                )

                t_id, c_id = (None, None)
                if template_and_cluster and vid in template_and_cluster:
                    t_id, c_id = template_and_cluster[vid]

                results.append(
                    EmbedResult(
                        variant_id=vid,
                        template_id=t_id,
                        cluster_id=c_id,
                        sequence_length=len(seq),
                        model_name=spec.model_name,
                        device=device,
                        embedding_path=str(npz_path.relative_to(outdir)),
                        status="ok",
                        error="",
                        elapsed_seconds=time.time() - start,
                    )
                )

        except Exception as e:
            # Mark each item in batch as failed
            for vid in batch_to_run:
                npz_path = _npz_path(outdir, spec, vid)
                t_id, c_id = (None, None)
                if template_and_cluster and vid in template_and_cluster:
                    t_id, c_id = template_and_cluster[vid]
                results.append(
                    EmbedResult(
                        variant_id=vid,
                        template_id=t_id,
                        cluster_id=c_id,
                        sequence_length=len(sequences[vid]),
                        model_name=spec.model_name,
                        device=device,
                        embedding_path=str(npz_path.relative_to(outdir)),
                        status="failed",
                        error=f"{type(e).__name__}: {e}",
                        elapsed_seconds=time.time() - start,
                    )
                )

        # flush every batch to be safe on HPC
        if results:
            _append_results(index_csv, results)
            results = []

    return []  # results already appended; return empty for now (wrapper will read index)
