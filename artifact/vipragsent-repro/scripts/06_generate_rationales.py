from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.annotation.prompts import RATIONALE_PROMPT, prompt_hash
from vipragsent.utils.azure_openai import azure_configured, chat_completion, first_message_content
from vipragsent.utils.env import load_env_file
from vipragsent.utils.io import read_jsonl, write_jsonl


def mock_rationale(record: dict) -> dict:
    return {
        "id": record["id"],
        "rationale": "Dry-run rationale placeholder. Human audit required before training use.",
        "generator": {
            "model": "local-mock",
            "decoding": "greedy",
            "prompt_hash": prompt_hash(RATIONALE_PROMPT),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "audit": {"sampled_for_audit": False, "faithful": None, "auditor": None},
    }


def azure_rationale(record: dict) -> dict:
    labels_json = json.dumps(record.get("labels", {}), ensure_ascii=False, sort_keys=True)
    response = chat_completion(
        [
            {"role": "system", "content": "You are a Vietnamese pragmatic-sentiment annotation assistant."},
            {
                "role": "user",
                "content": (
                    f"{RATIONALE_PROMPT}\n\n"
                    f"COMMENT: {record.get('text', '')}\n"
                    f"GOLD_LABELS: {labels_json}\n\n"
                    "Output only the Vietnamese explanation."
                ),
            },
        ],
        kind="rationale",
        temperature=0.0,
        top_p=1.0,
        max_tokens=220,
        json_mode=False,
    )
    return {
        "id": record["id"],
        "rationale": first_message_content(response).strip(),
        "generator": {
            "provider": "azure_openai",
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_RATIONALE", os.getenv("AZURE_OPENAI_DEPLOYMENT", "")),
            "decoding": "greedy",
            "prompt_hash": prompt_hash(RATIONALE_PROMPT),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": response,
        },
        "audit": {"sampled_for_audit": False, "faithful": None, "auditor": None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate rationale scaffold from adjudicated train labels.")
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "vipragsent_train.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "generated" / "rationales.jsonl"))
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI; otherwise create placeholders for pipeline testing only.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    if not Path(args.input).exists():
        print(json.dumps({"status": "pending_missing_gold_train", "input": args.input}, indent=2))
        return 0
    if args.azure and not azure_configured("rationale"):
        print(json.dumps({"status": "skipped_no_azure_openai_config", "message": "No rationale API call was made."}, indent=2))
        return 0

    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    generator = azure_rationale if args.azure else mock_rationale
    count = write_jsonl(args.output, (generator(record) for record in records))
    print(json.dumps({"status": "ok", "mode": "azure_openai" if args.azure else "mock", "records_written": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
