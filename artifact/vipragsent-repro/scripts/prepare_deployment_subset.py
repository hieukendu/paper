from __future__ import annotations

"""Create a deterministic fixed-budget training subset with provenance."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def counts(rows: list[dict]) -> dict[str, int]:
    return {label: sum(int(row["labels"][label]) for row in rows) for label in PRAGMATIC_LABELS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.input))
    if not 1 <= args.size <= len(rows):
        raise SystemExit(f"--size must be in [1, {len(rows)}]")
    # A single seeded uniform sample is deliberately simple: it avoids any
    # label-aware selection or test-informed curriculum hidden in this
    # deployment-budget reference.
    selected = random.Random(args.seed).sample(rows, args.size)
    selected.sort(key=lambda row: str(row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)
    metadata = {
        "method": "seeded_uniform_sample_without_replacement",
        "source": str(args.input),
        "source_sha256": digest(args.input),
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "seed": args.seed,
        "records": len(selected),
        "pragmatic_positive_counts": counts(selected),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
