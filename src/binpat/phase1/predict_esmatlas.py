"""
predict_esmatlas.py

Phase 1: predict structures for a set of sequences using the ESM Atlas folding API.

Responsibilities:
- Submit sequences to ESM Atlas fold endpoint
- Handle retries/backoff/timeouts
- Write resulting PDBs
- Return per-sequence status metadata for reporting
"""


from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal

import logging
import requests

from binpat.io.progress import Progress

logger = logging.getLogger(__name__)

DEFAULT_ESMATLAS_PDB_ENDPOINT = "https://api.esmatlas.com/foldSequence/v1/pdb/"


@dataclass(frozen=True)
class PredictionSpec:
    endpoint: str = DEFAULT_ESMATLAS_PDB_ENDPOINT
    timeout_seconds: float = 120.0

    # Retry policy
    max_retries: int = 6                 # total attempts = max_retries + 1
    backoff_base_seconds: float = 2.0    # exponential base multiplier
    backoff_max_seconds: float = 120.0   # cap sleep
    jitter_fraction: float = 0.2         # +/- 20% jitter on sleep time

    # Throttling
    sleep_between_requests: float = 0.2  # gentle default throttle

    # File behavior
    overwrite: bool = False              # if False: skip existing PDBs

    # Networking
    user_agent: str = "binpat/0.1.0 (ESM Atlas client)"


Status = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class PredictionResult:
    variant_id: str
    out_pdb_path: str
    status: Status                       # ok / skipped / failed
    reason: Optional[str]                # None for ok; "exists" for skipped; error message/code for failed
    status_code: Optional[int]
    attempts: int
    elapsed_seconds: float


def _headers(spec: PredictionSpec) -> Dict[str, str]:
    return {"User-Agent": spec.user_agent}


def _looks_like_pdb(text: str) -> bool:
    s = text.lstrip()
    return ("ATOM" in s) or ("HETATM" in s)


def _parse_retry_after_seconds(resp: requests.Response) -> Optional[float]:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def _sleep_with_jitter(base: float, *, rng: random.Random, jitter_fraction: float) -> None:
    if base <= 0:
        return
    jf = max(0.0, jitter_fraction)
    lo = base * (1.0 - jf)
    hi = base * (1.0 + jf)
    time.sleep(rng.uniform(lo, hi))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _post_sequence_for_pdb(seq: str, *, spec: PredictionSpec) -> requests.Response:
    return requests.post(
        spec.endpoint,
        data=seq.encode("utf-8"),
        headers=_headers(spec),
        timeout=spec.timeout_seconds,
    )


def predict_one(
    *,
    variant_id: str,
    sequence: str,
    out_pdb: Path,
    spec: PredictionSpec,
    rng: Optional[random.Random] = None,
) -> PredictionResult:
    """
    Predict one sequence. Handles retries, rate limits, and atomic writes.
    """
    start = time.time()
    rng = rng or random.Random(0)

    if out_pdb.exists() and not spec.overwrite:
        return PredictionResult(
            variant_id=variant_id,
            out_pdb_path=str(out_pdb),
            status="skipped",
            reason="exists",
            status_code=0,
            attempts=0,
            elapsed_seconds=0.0,
        )

    last_status: Optional[int] = None
    last_err: Optional[str] = None

    # gentle throttle between requests (helps avoid 429s)
    if spec.sleep_between_requests > 0:
        time.sleep(spec.sleep_between_requests)

    for attempt in range(spec.max_retries + 1):
        try:
            resp = _post_sequence_for_pdb(sequence, spec=spec)
            last_status = resp.status_code

            if resp.status_code == 200:
                body = resp.text
                if _looks_like_pdb(body):
                    _atomic_write_text(out_pdb, body)
                    return PredictionResult(
                        variant_id=variant_id,
                        out_pdb_path=str(out_pdb),
                        status="ok",
                        reason=None,
                        status_code=200,
                        attempts=attempt + 1,
                        elapsed_seconds=time.time() - start,
                    )
                last_err = f"bad_pdb_body_prefix={body[:200]!r}"

            elif resp.status_code == 429:
                ra = _parse_retry_after_seconds(resp)
                last_err = "rate_limited_429"
                if attempt < spec.max_retries:
                    sleep_s = ra if ra is not None else min(
                        spec.backoff_max_seconds,
                        spec.backoff_base_seconds * (2 ** attempt),
                    )
                    logger.warning(
                        "429 for %s; sleeping %.1fs (attempt %d/%d)",
                        variant_id,
                        sleep_s,
                        attempt + 1,
                        spec.max_retries + 1,
                    )
                    _sleep_with_jitter(sleep_s, rng=rng, jitter_fraction=spec.jitter_fraction)
                    continue

            elif 500 <= resp.status_code < 600:
                last_err = f"server_error_{resp.status_code}"

            else:
                # likely non-retriable
                last_err = f"http_{resp.status_code}_prefix={resp.text[:200]!r}"
                break

        except requests.Timeout as e:
            last_err = f"timeout: {e}"
        except requests.RequestException as e:
            last_err = f"request_exception: {type(e).__name__}: {e}"

        # Retry for transient errors (429 / 5xx / request exceptions)
        should_retry = (
            attempt < spec.max_retries
            and (
                last_status is None
                or last_status == 429
                or (last_status is not None and 500 <= last_status < 600)
            )
        )
        if should_retry:
            sleep_s = min(spec.backoff_max_seconds, spec.backoff_base_seconds * (2 ** attempt))
            logger.warning(
                "Retrying %s after %.1fs (attempt %d/%d). status=%s err=%s",
                variant_id,
                sleep_s,
                attempt + 1,
                spec.max_retries + 1,
                last_status,
                last_err,
            )
            _sleep_with_jitter(sleep_s, rng=rng, jitter_fraction=spec.jitter_fraction)
            continue

        break

    return PredictionResult(
        variant_id=variant_id,
        out_pdb_path=str(out_pdb),
        status="failed",
        reason=last_err or "unknown_error",
        status_code=last_status,
        attempts=spec.max_retries + 1,
        elapsed_seconds=time.time() - start,
    )


def predict_many(
    *,
    sequences: Dict[str, str],
    out_dir: Path,
    spec: PredictionSpec,
    rng_seed: int = 0,
) -> List[PredictionResult]:
    """
    Predict a batch of sequences serially. Writes to out_dir/<variant_id>.pdb
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)

    p = Progress(total=len(sequences), label="structures downloaded")
    results: List[PredictionResult] = []
    num = 0
    for vid, seq in sequences.items():
        out_pdb = out_dir / f"{vid}.pdb"
        out_pdb = out_pdb.replace("|", "_")
        results.append(
            predict_one(
                variant_id=vid,
                sequence=seq,
                out_pdb=out_pdb,
                spec=spec,
                rng=rng,
            )
        )
        num += 1
        p.update(num)
    return results


def failed_ids(results: List[PredictionResult]) -> List[str]:
    """Return variant_ids that failed prediction (status == 'failed')."""
    return [r.variant_id for r in results if r.status == "failed"]


def write_failed_ids_txt(results: List[PredictionResult], out_path: Path) -> None:
    """Write ids_failed.txt (one variant_id per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = failed_ids(results)
    out_path.write_text("\n".join(ids) + ("\n" if ids else ""))
