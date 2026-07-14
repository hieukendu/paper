from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.artifacts.make_artifacts import make_artifacts_dryrun


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pending artifact schemas and claim ledger only.")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    report = make_artifacts_dryrun(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
