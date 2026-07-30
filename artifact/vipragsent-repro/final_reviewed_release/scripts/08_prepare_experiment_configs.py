from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.utils.io import dump_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the active experiment-config plan without running training.")
    parser.add_argument("--output", default=str(ROOT / "results" / "pending" / "experiment_command_plan.json"))
    args = parser.parse_args()

    plan = {
        "status": "scaffold_only",
        "warning": "Do not execute training commands until HUMAN_ACTIONS.md is complete.",
        "commands": [
            {"experiment": "q1", "config": "configs/experiments_q1.yaml", "training": "deferred"},
            {"experiment": "q2", "config": "configs/experiments_q2.yaml", "training": "deferred"},
            {"experiment": "p0_p1_p2", "config": "configs/experiments_p0_p1_p2.yaml", "training": "artifact_validation_only"},
            {"experiment": "q4", "config": "configs/experiments_q4.yaml", "training": "deferred"},
        ],
    }
    dump_json(args.output, plan)
    print(json.dumps({"status": "ok", "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
