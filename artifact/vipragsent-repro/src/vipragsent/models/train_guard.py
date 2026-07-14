from __future__ import annotations

import os
import subprocess

from vipragsent.utils.env import env_bool, env_float


class TrainingBlocked(RuntimeError):
    """Raised when a training run violates setup-only guardrails."""


def detect_vram_gb() -> float:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            values = [float(line.strip()) / 1024 for line in proc.stdout.splitlines() if line.strip()]
            if values:
                return max(values)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return env_float("LOCAL_GPU_VRAM_GB", 0.0)


def assert_training_allowed(*, confirm_run_training: bool, allow_low_vram: bool = False) -> None:
    if not env_bool("ENABLE_FINE_TUNING", False):
        raise TrainingBlocked(
            "Fine-tuning is disabled. Set ENABLE_FINE_TUNING=true and pass "
            "--confirm-run-training only after human approval."
        )
    if os.getenv("RUN_MODE", "setup_only").strip().lower() == "setup_only":
        raise TrainingBlocked("RUN_MODE=setup_only; refusing to train.")
    if not confirm_run_training:
        raise TrainingBlocked("Refusing to train without --confirm-run-training.")
    vram_gb = detect_vram_gb()
    if vram_gb and vram_gb < 8 and not allow_low_vram:
        raise TrainingBlocked("Detected low VRAM. Use cloud GPU or pass --allow-low-vram for toy runs only.")
