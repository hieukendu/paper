from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.schema import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.utils.azure_openai import AzureOpenAIError, azure_configured, chat_completion, first_message_content
from vipragsent.utils.env import load_env_file
from vipragsent.utils.io import read_jsonl, write_jsonl


SYSTEM_PROMPT = """You are a Vietnamese pragmatic-sentiment classifier.
Predict six binary pragmatic labels, intended polarity, and UIT-VSMEC emotion.
Return JSON only. Do not include explanations."""


def main() -> int:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Azure OpenAI no-fine-tuning baseline predictions.")
    parser.add_argument("--input", required=True, help="JSONL records to classify.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["zero_shot", "8_shot"], default="zero_shot")
    parser.add_argument("--demo-source", help="Adjudicated train JSONL for 8-shot demonstrations.")
    parser.add_argument("--system-prefix", default=os.getenv("VIPRAGSENT_API_SYSTEM_PREFIX", "gpt41_mini"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate-limit-seconds", type=float, default=0.0)
    parser.add_argument("--confirm-api-call", action="store_true")
    args = parser.parse_args()

    if not args.confirm_api_call:
        print(json.dumps({"status": "blocked_confirmation_required", "message": "Pass --confirm-api-call to use Azure OpenAI."}, indent=2))
        return 0
    if not azure_configured("label"):
        print(json.dumps({"status": "blocked_missing_azure_config"}, indent=2))
        return 0

    demos = []
    if args.mode == "8_shot":
        if not args.demo_source:
            raise SystemExit("--demo-source is required for --mode 8_shot")
        demos = _select_demos(list(read_jsonl(args.demo_source)), n=8)

    output_path = Path(args.output)
    existing_records = list(read_jsonl(output_path)) if output_path.exists() else []
    existing_ids = {str(record.get("id")) for record in existing_records}

    outputs = []
    errors = []
    system_id = f"{args.system_prefix}_{args.mode}"
    for idx, record in enumerate(read_jsonl(args.input), start=1):
        if args.limit is not None and len(outputs) >= args.limit:
            break
        if str(record.get("id")) in existing_ids:
            continue
        try:
            predictions, probabilities, raw_response = _predict_one(record, demos=demos)
            outputs.append(
                {
                    "id": record["id"],
                    "system": system_id,
                    "seed": 20260520,
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "provider": "azure_openai",
                    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_LABEL", os.getenv("AZURE_OPENAI_DEPLOYMENT", "")),
                    "raw_response": raw_response,
                }
            )
        except (AzureOpenAIError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"id": str(record.get("id")), "error": str(exc)})
            if len(errors) >= 5:
                break
        if args.rate_limit_seconds:
            time.sleep(args.rate_limit_seconds)
        if idx % 25 == 0:
            write_jsonl(output_path, [*existing_records, *outputs])
    count = write_jsonl(output_path, [*existing_records, *outputs])
    status = "ok" if not errors else "partial_failed"
    print(
        json.dumps(
            {"status": status, "output": args.output, "system": system_id, "records": count, "errors": errors},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _predict_one(record: dict[str, Any], *, demos: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in demos:
        messages.append({"role": "user", "content": _comment_prompt(demo, include_schema=False)})
        messages.append({"role": "assistant", "content": json.dumps({"labels": demo["labels"]}, ensure_ascii=False)})
    messages.append({"role": "user", "content": _comment_prompt(record, include_schema=True)})
    raw = chat_completion(messages, kind="label", temperature=0.0, top_p=1.0, max_tokens=700, json_mode=True)
    parsed = json.loads(first_message_content(raw))
    labels = canonicalize_labels(parsed.get("labels", parsed))
    _validate_labels(labels)
    probabilities = _normalise_probabilities(parsed.get("probabilities") or parsed.get("probs") or {})
    return labels, probabilities, raw


def _comment_prompt(record: dict[str, Any], *, include_schema: bool) -> str:
    text = record.get("text_normalized") or record.get("text") or ""
    prompt = f"Comment:\n{text}\n"
    if include_schema:
        prompt += (
            "\nReturn JSON with a top-level labels object containing exactly: "
            f"{', '.join(PRAGMATIC_LABELS)}, polarity, emotion. "
            "Use 0/1 for pragmatic labels. "
            f"Allowed polarity: {sorted(POLARITY_LABELS)}. "
            f"Allowed emotion: {sorted(EMOTION_LABELS)}. "
            "Optionally include probabilities for calibration."
        )
    return prompt


def _select_demos(records: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    selected = []
    covered = set()
    for label in PRAGMATIC_LABELS:
        for record in records:
            labels = canonicalize_labels(record.get("labels"))
            if labels.get(label) in (1, True) and record.get("id") not in covered:
                selected.append({"text": record.get("text"), "labels": labels})
                covered.add(record.get("id"))
                break
        if len(selected) >= n:
            return selected[:n]
    for record in records:
        if record.get("id") in covered:
            continue
        selected.append({"text": record.get("text"), "labels": canonicalize_labels(record.get("labels"))})
        if len(selected) >= n:
            return selected[:n]
    return selected[:n]


def _validate_labels(labels: dict[str, Any]) -> None:
    for field in PRAGMATIC_LABELS:
        labels[field] = 1 if labels.get(field) in (1, True, "1", "true", "True") else 0
    if labels.get("polarity") not in POLARITY_LABELS:
        raise ValueError(f"invalid polarity: {labels.get('polarity')}")
    if labels.get("emotion") not in EMOTION_LABELS:
        raise ValueError(f"invalid emotion: {labels.get('emotion')}")


def _normalise_probabilities(probabilities: Any) -> dict[str, Any]:
    return probabilities if isinstance(probabilities, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
