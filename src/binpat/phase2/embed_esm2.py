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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


DEFAULT_MODEL = "esm2_t36_3B_UR50D"
DEFAULT_LAYER = 36
DEFAULT_DIM = 2560


@dataclass(frozen=True)
class EmbedSpec:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 1
    device: str = "auto"  # auto | cpu | cuda | mps
    dtype: str = "float32"  # float32 (matches newer pipeline) | float16
    overwrite: bool = False

    out_dirname: str = "phase2"
    embeddings_subdir: str = "embeddings"
    index_filename: str = "embeddings_index.csv"

    # Fair-ESM layer index; 36 is the final transformer layer of the 3B model.
    layer: int = DEFAULT_LAYER

    # Optional *total token* count, including BOS and EOS, as with HF tokenizer.
    # None: no truncation; max_length=128 retains up to 126 residues.
    # Truncation creates an embedding shorter than the stored original sequence.
    max_length: Optional[int] = None


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


def _validate_spec(spec: EmbedSpec) -> None:
    if spec.model_name != DEFAULT_MODEL:
        raise ValueError(
            f"This reproducibility version supports {DEFAULT_MODEL} only; "
            f"got {spec.model_name!r}."
        )
    if not isinstance(spec.layer, int) or not 0 <= spec.layer <= DEFAULT_LAYER:
        raise ValueError(f"layer must be an integer in 0..{DEFAULT_LAYER}")
    if int(spec.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if spec.max_length is not None and spec.max_length < 3:
        raise ValueError("max_length must be >= 3 (includes BOS and EOS)")
    _resolve_dtype(spec.dtype)


def _resolve_device(device: str) -> torch.device:
    d = (device or "auto").strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            d = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            d = "mps"
        else:
            d = "cpu"
    if d not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {device!r}; choose auto, cpu, cuda, or mps")
    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if d == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return torch.device(d)


def _resolve_dtype(dtype: str) -> np.dtype:
    try:
        result = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"Unsupported output dtype: {dtype!r}") from exc
    if result not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("dtype must be float16 or float32")
    return result


def _npz_path(outdir: Path, spec: EmbedSpec, vid: str) -> Path:
    return outdir / spec.out_dirname / spec.embeddings_subdir / f"{vid}.npz"


def _append_results(index_csv: Path, rows: List[EmbedResult]) -> None:
    if not rows:
        return
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    has_header = index_csv.exists() and index_csv.stat().st_size > 0
    with index_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        if not has_header:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _effective_length(seq: str, max_length: Optional[int]) -> int:
    return len(seq) if max_length is None else min(len(seq), max_length - 2)


def _matching_npz(path: Path, seq: str, spec: EmbedSpec) -> bool:
    """Skip only validated fair-esm 3B NPZs, never arbitrary preexisting files."""
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "emb", "seq", "model_name", "representation_layer",
                "embedding_dim", "implementation",
            }
            if not required.issubset(data.files):
                return False
            emb = data["emb"]
            return (
                emb.shape == (_effective_length(seq, spec.max_length), DEFAULT_DIM)
                and emb.dtype == _resolve_dtype(spec.dtype)
                and np.isfinite(emb).all()
                and str(data["seq"].item()) == seq
                and str(data["model_name"].item()) == spec.model_name
                and int(data["representation_layer"].item()) == spec.layer
                and int(data["embedding_dim"].item()) == DEFAULT_DIM
                and str(data["implementation"].item()) == "fair-esm"
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def _load_model(spec: EmbedSpec, device: torch.device):
    import esm  # pip install fair-esm; not Hugging Face transformers

    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model.eval()
    model = model.to(device)

    if getattr(model, "num_layers", DEFAULT_LAYER) != DEFAULT_LAYER:
        raise RuntimeError("Loaded ESM-2 model has an unexpected transformer depth")
    if getattr(model, "embed_dim", DEFAULT_DIM) != DEFAULT_DIM:
        raise RuntimeError("Loaded ESM-2 model has an unexpected embedding dimension")

    return model, alphabet


def _save_npz(path: Path, *, emb: np.ndarray, seq: str, spec: EmbedSpec) -> None:
    """Use atomic replacement so interrupted runs do not leave partial NPZs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                emb=emb,
                seq=np.array([seq]),                # original shape preserved
                model_name=np.array([spec.model_name]),
                representation_layer=np.array([spec.layer], dtype=np.int32),
                embedding_dim=np.array([DEFAULT_DIM], dtype=np.int32),
                implementation=np.array(["fair-esm"]),
            )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def embed_and_save_many(
    *,
    sequences: Dict[str, str],
    outdir: Path,
    spec: EmbedSpec,
    template_and_cluster: Optional[
        Dict[str, Tuple[Optional[str], Optional[int]]]
    ] = None,
) -> List[EmbedResult]:
    """Embed sequences with fair-esm and write one NPZ per variant.

    (an empty list; statuses are appended
    to embeddings_index.csv after each batch). Failures are logged per variant.
    """
    _validate_spec(spec)
    outdir = Path(outdir)
    emb_dir = outdir / spec.out_dirname / spec.embeddings_subdir
    index_csv = outdir / spec.out_dirname / spec.index_filename
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(spec.device)
    np_dtype = _resolve_dtype(spec.dtype)

    vids = list(sequences)
    pending: List[str] = []
    skipped: List[EmbedResult] = []

    def make_result(vid: str, status: str, error: str, elapsed: float) -> EmbedResult:
        t_id, c_id = (None, None)
        if template_and_cluster and vid in template_and_cluster:
            t_id, c_id = template_and_cluster[vid]
        return EmbedResult(
            variant_id=vid,
            template_id=t_id,
            cluster_id=c_id,
            sequence_length=len(sequences[vid]),
            model_name=spec.model_name,
            device=str(device),
            embedding_path=str(_npz_path(outdir, spec, vid).relative_to(outdir)),
            status=status,
            error=error,
            elapsed_seconds=elapsed,
        )

    for vid in vids:
        seq = sequences[vid]
        path = _npz_path(outdir, spec, vid)
        if not spec.overwrite and path.exists():
            if _matching_npz(path, seq, spec):
                skipped.append(make_result(vid, "skipped", "exists_valid", 0.0))
            else:
                raise RuntimeError(
                    f"Incompatible preexisting embedding: {path}. "
                    "It may be an old Hugging Face / 650M embedding. "
                    "Use a fresh outdir or set EmbedSpec(overwrite=True) "
                    "to regenerate intentionally."
                )
        else:
            pending.append(vid)

    _append_results(index_csv, skipped)
    if not pending:
        print(f"[embedding] All {len(vids)} sequences already have valid embeddings.")
        return []

    model, alphabet = _load_model(spec, device)
    # Fair-esm uses an Alphabet-based batch converter (BOS/EOS/padding),
    # max_length is total tokens incl two specials.
    residue_limit = None if spec.max_length is None else spec.max_length - 2
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=residue_limit)

    print(
        f"[embedding] model={spec.model_name}, layer={spec.layer}, "
        f"device={device}, storage={np_dtype}, to_embed={len(pending)}, "
        f"already_valid={len(skipped)}"
    )
    if spec.max_length is not None:
        print(
            "[embedding] WARNING: max_length includes two special tokens; "
            f"sequences longer than {residue_limit} residues will be truncated. "
            "The saved 'seq' field retains the original full sequence."
        )

    for start in range(0, len(pending), spec.batch_size):
        batch_vids = pending[start : start + spec.batch_size]
        start_time = time.time()
        results: List[EmbedResult] = []
        try:
            batch = [(vid, sequences[vid]) for vid in batch_vids]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            with torch.no_grad():
                outputs = model(
                    tokens,
                    repr_layers=[spec.layer],
                    return_contacts=False,
                )
                hidden = outputs["representations"][spec.layer]

            if hidden.ndim != 3 or hidden.shape[0] != len(batch_vids):
                raise RuntimeError("Unexpected ESM-2 representation tensor shape")

            # 0 = BOS; positions 1..L are amino acids; EOS and padding follow.
            # Slicing by each sequence length avoids including EOS/padding.
            for bi, vid in enumerate(batch_vids):
                seq = sequences[vid]
                L = _effective_length(seq, spec.max_length)
                reps = hidden[bi, 1 : L + 1]
                if tuple(reps.shape) != (L, DEFAULT_DIM):
                    raise RuntimeError(
                        f"{vid}: expected {(L, DEFAULT_DIM)}; got {tuple(reps.shape)}"
                    )
                # Cast to float32 on CPU before optionally saving float16.
                emb = reps.detach().to(device="cpu", dtype=torch.float32).numpy()
                emb = emb.astype(np_dtype, copy=False)
                if not np.isfinite(emb).all():
                    raise RuntimeError(f"{vid}: nonfinite embedding detected")
                _save_npz(_npz_path(outdir, spec, vid), emb=emb, seq=seq, spec=spec)
                results.append(make_result(vid, "ok", "", time.time() - start_time))

            del tokens, outputs, hidden
        except Exception as exc:
            # Mark only unsaved variants as failed if a batch partially completed.
            successes = {r.variant_id for r in results}
            for vid in batch_vids:
                if vid not in successes:
                    results.append(
                        make_result(vid, "failed", f"{type(exc).__name__}: {exc}",
                                    time.time() - start_time)
                    )
        finally:
            _append_results(index_csv, results)

        print(f"[embedding] processed {min(start + spec.batch_size, len(pending))}/{len(pending)}")

    return []
