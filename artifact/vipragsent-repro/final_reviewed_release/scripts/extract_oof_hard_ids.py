"""Export audited OOF error IDs for train-only targeted reweighting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--error-type", choices=("fn", "fp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["error_type"] == args.error_type:
                ids.append(item["id"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("expected a non-empty unique OOF ID list")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sorted(ids), indent=2) + "\n")
    print(json.dumps({"status": "ok", "error_type": args.error_type, "ids": len(ids), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
