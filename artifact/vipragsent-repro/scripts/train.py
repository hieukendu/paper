from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.models.deferred_train import deferred_training_message
from vipragsent.models.train_guard import TrainingBlocked, assert_training_allowed
from vipragsent.utils.env import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard-railed training entrypoint. Disabled by default.")
    parser.add_argument("--config", default="configs/training_local.yaml")
    parser.add_argument("--confirm-run-training", action="store_true")
    parser.add_argument("--allow-low-vram", action="store_true", help="For toy smoke runs only after approval.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    try:
        assert_training_allowed(confirm_run_training=args.confirm_run_training, allow_low_vram=args.allow_low_vram)
    except TrainingBlocked as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(deferred_training_message())


if __name__ == "__main__":
    raise SystemExit(main())
