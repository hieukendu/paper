from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.artifacts.reporting import make_final_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tables, SVG figures, and claim ledger from results/*.json.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--tables-dir", default=str(ROOT / "tables"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    parser.add_argument("--claim-ledger", default=str(ROOT / "results" / "claim_ledger.csv"))
    args = parser.parse_args()
    summary = make_final_artifacts(
        root=ROOT,
        results_dir=args.results_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        claim_ledger=args.claim_ledger,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
