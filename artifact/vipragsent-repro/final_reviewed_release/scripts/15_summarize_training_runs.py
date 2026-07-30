from __future__ import annotations

"""Turn trainer histories/manifests into Q4 learning and measured-cost inputs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--predictions", default=str(ROOT / "results" / "predictions"))
    args = parser.parse_args()
    output_root, prediction_root = Path(args.outputs), Path(args.predictions)
    curves: dict[str, list[dict[str, object]]] = defaultdict(list)
    cost_systems: dict[str, dict[str, object]] = defaultdict(lambda: {"runs": [], "gpu_hours": 0.0})
    for manifest_path in sorted(output_root.glob("**/run_manifest.json")):
        manifest = _load(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("status") != "ok":
            continue
        system = str(manifest.get("system"))
        run_dir = manifest_path.parent
        history_path = run_dir / "history.json"
        if history_path.exists():
            history = _load(history_path)
            curves[system].append({"seed": manifest.get("seed"), "history": history, "run_manifest": str(manifest_path)})
        elapsed = float(manifest.get("elapsed_seconds") or 0.0)
        cost_systems[system]["runs"].append(str(manifest_path))
        cost_systems[system]["gpu_hours"] = round(float(cost_systems[system]["gpu_hours"]) + elapsed / 3600.0, 6)
    learning_path = prediction_root / "learning_curves" / "summary.json"
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.write_text(json.dumps({"curves": curves}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cost_path = prediction_root / "run_costs.json"
    cost_path.write_text(json.dumps({"source": "trainer run_manifest.json elapsed_seconds", "systems": cost_systems}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "learning_curves": str(learning_path), "run_costs": str(cost_path), "systems": sorted(cost_systems)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
