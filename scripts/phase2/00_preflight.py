#!/usr/bin/env python3
"""
Phase 2 preflight: verify GPU + key Python deps for embedding/prediction steps.

Typical use:
  python scripts/phase2/00_preflight.py
  python scripts/phase2/00_preflight.py --require-gpu
  python scripts/phase2/00_preflight.py --require-libs torch,transformers,fair_esm,accelerate
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, (p.stdout or "").strip()
    except Exception as e:
        return 1, f"<failed to run {' '.join(cmd)}: {e}>"


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _try_import(modname: str):
    try:
        return importlib.import_module(modname), None
    except Exception as e:
        return None, e


def _print_kv(title: str, kv: Dict[str, str]) -> None:
    print(f"\n== {title} ==")
    if not kv:
        print("(none)")
        return
    w = max(len(k) for k in kv.keys())
    for k, v in kv.items():
        print(f"{k:<{w}} : {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 preflight (GPU + deps)")
    p.add_argument("--require-gpu", action="store_true", help="Exit nonzero if CUDA is not available.")
    p.add_argument(
        "--require-libs",
        type=str,
        default="",
        help="Comma-separated module names to require import (e.g., torch,transformers,fair_esm,accelerate).",
    )
    p.add_argument(
        "--print-env",
        action="store_true",
        help="Print common environment variables related to CUDA/HF caches.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Basic system info ---
    _print_kv(
        "System",
        {
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cwd": os.getcwd(),
        },
    )

    # --- NVIDIA tools presence ---
    nvidia_smi = _which("nvidia-smi")
    _print_kv("GPU tools", {"nvidia-smi": nvidia_smi or "(not found)"})

    if nvidia_smi:
        rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        print("\n== nvidia-smi (GPU summary) ==")
        if rc == 0 and out:
            print(out)
        else:
            # fallback to plain nvidia-smi
            rc2, out2 = _run(["nvidia-smi"])
            print(out2 if out2 else out)

    # --- Torch / CUDA check ---
    torch, torch_err = _try_import("torch")
    if torch is None:
        print("\n== torch ==")
        print(f"IMPORT FAILED: {torch_err}")
        if args.require_gpu or ("torch" in args.require_libs.split(",") if args.require_libs else False):
            return 2
    else:
        cuda_ok = bool(torch.cuda.is_available())
        torch_info = {
            "torch.__version__": getattr(torch, "__version__", "?"),
            "torch.version.cuda": str(getattr(torch.version, "cuda", None)),
            "cuda_available": str(cuda_ok),
            "device_count": str(torch.cuda.device_count()),
        }
        if cuda_ok and torch.cuda.device_count() > 0:
            try:
                torch_info["gpu_0_name"] = str(torch.cuda.get_device_name(0))
            except Exception as e:
                torch_info["gpu_0_name"] = f"<error: {e}>"

        # Extra info that helps debug mismatches
        try:
            torch_info["torch.backends.cudnn.version"] = str(torch.backends.cudnn.version())
        except Exception:
            torch_info["torch.backends.cudnn.version"] = "(unavailable)"

        _print_kv("torch / CUDA", torch_info)

        if args.require_gpu and not cuda_ok:
            print("\nERROR: --require-gpu set but torch.cuda.is_available() is False.")
            return 3

    # --- Optional libs (import-only) ---
    required = [s.strip() for s in args.require_libs.split(",") if s.strip()]
    if required:
        print("\n== Required imports ==")
        failures = 0
        for name in required:
            mod, err = _try_import(name)
            if mod is None:
                failures += 1
                print(f"[FAIL] {name}: {err}")
            else:
                ver = getattr(mod, "__version__", None)
                if ver is None and name == "fair_esm":
                    # Some installs expose esm as the import name
                    ver = getattr(mod, "version", None)
                print(f"[ OK ] {name}" + (f" (version={ver})" if ver else ""))

        if failures:
            print(f"\nERROR: {failures} required import(s) failed.")
            return 4

    # --- Common Phase 2 deps ---
    candidates = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
        ("tqdm", "tqdm"),
        # ESM can appear as `esm` (fair-esm) depending on how it is installed
        ("esm", "esm"),
        ("fair_esm", "fair_esm"),
    ]

    print("\n== Optional imports (best-effort) ==")
    for label, modname in candidates:
        mod, err = _try_import(modname)
        if mod is None:
            print(f"[ -- ] {label}: not importable ({err.__class__.__name__})")
        else:
            ver = getattr(mod, "__version__", None)
            print(f"[ OK ] {label}" + (f" (version={ver})" if ver else ""))

    # --- Environment vars helpful on clusters ---
    if args.print_env:
        keys = [
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_GPUS",
            "SLURM_GPUS_ON_NODE",
            "SLURM_CPUS_PER_TASK",
            "SLURM_TMPDIR",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "HF_HUB_ENABLE_HF_TRANSFER",
        ]
        env = {k: os.environ.get(k, "") for k in keys if os.environ.get(k) is not None}
        _print_kv("Environment", env)

    print("\nPreflight OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

