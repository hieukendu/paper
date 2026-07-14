from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.gold_builder import build_gold_after_annotation
from vipragsent.artifacts.reporting import make_final_artifacts
from vipragsent.experiments.runner import run_no_finetune_suite
from vipragsent.utils.env import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-annotation no-fine-tuning pipeline: build gold splits, optionally run API baselines, evaluate, and make artifacts."
    )
    parser.add_argument("--run-api-baselines", action="store_true")
    parser.add_argument("--confirm-api-call", action="store_true")
    parser.add_argument("--api-limit", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--allow-silver-gold", action="store_true", help="Smoke-test only.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    gold_report = build_gold_after_annotation(
        batches_dir=ROOT / "data" / "annotation" / "batches",
        reviewer_1_dir=ROOT / "data" / "annotation" / "reviewer_01",
        reviewer_2_dir=ROOT / "data" / "annotation" / "reviewer_02",
        adjudicated_dir=ROOT / "data" / "annotation" / "adjudicated",
        output_dir=ROOT / "data" / "processed",
        disagreements_output=ROOT / "data" / "annotation" / "disagreements" / "all_disagreements.jsonl",
        report_output=ROOT / "data" / "manifest" / "gold_build_report.json",
    )
    api_outputs = []
    if args.run_api_baselines:
        if not args.confirm_api_call:
            api_outputs.append({"status": "blocked_confirmation_required", "message": "Pass --confirm-api-call to run API baselines."})
        else:
            test_path = ROOT / "data" / "processed" / "vipragsent_test.jsonl"
            train_path = ROOT / "data" / "processed" / "vipragsent_train.jsonl"
            for mode, system_id in [
                ("zero_shot", "gpt41_mini_zero_shot"),
                ("8_shot", "gpt41_mini_8_shot"),
            ]:
                output_path = ROOT / "results" / "predictions" / "main_pragmatic" / system_id / "20260520.jsonl"
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "predict_openai_baseline.py"),
                    "--input",
                    str(test_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    mode,
                    "--confirm-api-call",
                ]
                if mode == "8_shot":
                    command.extend(["--demo-source", str(train_path)])
                if args.api_limit is not None:
                    command.extend(["--limit", str(args.api_limit)])
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                api_outputs.append(
                    {
                        "mode": mode,
                        "returncode": completed.returncode,
                        "stdout": _json_or_text(completed.stdout),
                        "stderr": completed.stderr.strip(),
                    }
                )

    experiment_summary = run_no_finetune_suite(
        root=ROOT,
        vipragsent_test=ROOT / "data" / "processed" / "vipragsent_test.jsonl",
        public_data=ROOT / "data" / "processed" / "all_unified.jsonl",
        bootstrap_resamples=args.bootstrap_resamples,
        allow_silver_gold=args.allow_silver_gold,
    )
    artifact_summary = make_final_artifacts(root=ROOT)
    print(
        json.dumps(
            {
                "status": "ok",
                "gold": gold_report,
                "api_baselines": api_outputs,
                "experiments": experiment_summary,
                "artifacts": artifact_summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _json_or_text(value: str):
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
