from __future__ import annotations

import os
import re
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vipragsent.annotation.prompts import PRAGMATIC_LABELING_PROMPT, prompt_hash
from vipragsent.data.schema import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.utils.azure_openai import AzureOpenAIError, azure_configured, chat_completion, first_message_content
from vipragsent.utils.io import read_jsonl, write_jsonl

ENGLISH_TOKEN_RE = re.compile(r"\b(vibe|gg|crush|deadline|trend|best|fail|okela|idol)\b", re.IGNORECASE)


def heuristic_silver_labels(text: str) -> dict[str, Any]:
    lowered = text.lower()
    sarcasm = int(any(marker in lowered for marker in ["=))", ":))", "hay qua", "hay thi", "dinh that"]))
    mocking = int(any(term in lowered for term in ["nasa", "thien tai", "qua gioi", "dinh cao"]) or sarcasm)
    code_switching = int(bool(ENGLISH_TOKEN_RE.search(text)))
    idiom = int(any(term in lowered for term in ["di vao long dat", "ngan nam", "con chi chi"]))
    irony = int(sarcasm or "5 sao" in lowered)
    implicit = int(any([sarcasm, irony, idiom, mocking]) and not any(word in lowered for word in ["te", "xau", "tot"]))
    polarity = "negative" if sarcasm or mocking else "neutral"
    emotion = "disgust" if sarcasm or mocking else "other"
    return canonicalize_labels(
        {
            "implicit_sentiment": implicit,
            "sarcasm": sarcasm,
            "irony": irony,
            "idiom_figurative": idiom,
            "code_switching": code_switching,
            "mocking": mocking,
            "polarity": polarity,
            "emotion": emotion,
        }
    )


def local_reasoning_rationale(text: str, labels: dict[str, Any]) -> str:
    lowered = text.lower()
    cues: list[str] = []
    if labels.get("sarcasm") in (1, True):
        if any(marker in lowered for marker in ["=))", ":))", "=)))", " :))", " :)))"]):
            cues.append("dấu mặt cười/mỉa mai như '=))' hoặc ':))'")
        if any(term in lowered for term in ["hay qua", "dinh", "quá nhanh", "quá nguy hiểm", "tuyệt vời", "đỉnh"]):
            cues.append("lời khen hoặc phóng đại dễ đảo nghĩa")
    if labels.get("mocking") in (1, True):
        cues.append("giọng châm chọc/chế giễu một người, hành vi hoặc tình huống")
    if labels.get("code_switching") in (1, True):
        cues.append("có từ tiếng Anh/slang như signal code-switching")
    if labels.get("idiom_figurative") in (1, True):
        cues.append("cụm từ mang nghĩa bóng hoặc thành ngữ")
    if labels.get("irony") in (1, True):
        cues.append("có sự lệch giữa kỳ vọng và cách nói bề mặt")
    if labels.get("implicit_sentiment") in (1, True):
        cues.append("sentiment không nói thẳng mà ẩn trong ngữ cảnh/cách nói")
    if labels.get("polarity") == "negative":
        polarity_text = "ý định tổng thể nghiêng về tiêu cực"
    elif labels.get("polarity") == "positive":
        polarity_text = "ý định tổng thể nghiêng về tích cực"
    else:
        polarity_text = "ý định tổng thể trung tính"
    emotion = labels.get("emotion") or "other"
    if cues:
        reason = " và ".join(cues[:2])
        return f"Câu này được gán nhãn như vậy vì {reason}, nên {polarity_text} và cảm xúc phù hợp là {emotion}."
    return f"Câu này không có cue dụng học quá rõ nên được giữ theo nhãn bề mặt, với {polarity_text} và cảm xúc {emotion}."


def azure_silver_labels(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    user_prompt = f"""Comment:
{text}

Return JSON with exactly this shape:
{{
  "labels": {{
    "implicit_sentiment": 0,
    "sarcasm": 0,
    "irony": 0,
    "idiom_figurative": 0,
    "code_switching": 0,
    "mocking": 0,
    "polarity": "neutral",
    "emotion": "other"
  }},
  "rationale_short": "one short Vietnamese reason"
}}

Allowed polarity values: {sorted(POLARITY_LABELS)}
Allowed emotion values: {sorted(EMOTION_LABELS)}
Use 0/1 integers for pragmatic labels. These are silver labels only."""
    raw = chat_completion(
        [
            {"role": "system", "content": PRAGMATIC_LABELING_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        kind="label",
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
        json_mode=True,
    )
    content = first_message_content(raw)
    parsed = json.loads(content)
    label_payload = parsed.get("labels", parsed)
    labels = canonicalize_labels(label_payload)
    for field in PRAGMATIC_LABELS:
        labels[field] = 1 if labels.get(field) in (1, True, "1", "true", "True") else 0
    if labels.get("polarity") not in POLARITY_LABELS:
        labels["polarity"] = None
    if labels.get("emotion") not in EMOTION_LABELS:
        labels["emotion"] = None
    return labels, {"parsed_response": parsed, "raw_response": raw}


def prelabel_record(
    record: dict[str, Any],
    *,
    model_id: str,
    provider: str = "local_heuristic",
    use_azure: bool = False,
) -> dict[str, Any]:
    if use_azure:
        labels, response_payload = azure_silver_labels(record.get("text", ""))
    else:
        labels = heuristic_silver_labels(record.get("text", ""))
        response_payload = {"mode": "mock_silver_labels", "labels": labels}
    output = dict(record)
    output["labels"] = labels
    output["available_labels"] = {"pragmatic": True, "polarity": True, "emotion": True}
    output["label_status"] = "silver_agent"
    output["agent"] = {
        "provider": provider,
        "model": model_id,
        "prompt_hash": prompt_hash(PRAGMATIC_LABELING_PROMPT),
        "temperature": 0.0,
        "top_p": 1.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": response_payload,
    }
    output["rationale"] = local_reasoning_rationale(record.get("text", ""), labels)
    output.setdefault("annotation", {})
    output["annotation"]["needs_human_review"] = True
    output["agent"]["reasoning"] = output["rationale"]
    return output


def run_prelabelling(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mock: bool = False,
    use_azure: bool = False,
    limit: int | None = None,
    rate_limit_seconds: float = 0.0,
) -> dict[str, Any]:
    if use_azure and not azure_configured("label"):
        return {
            "status": "skipped_no_azure_openai_config",
            "message": "Azure OpenAI env vars are missing; no API call was made.",
            "output": str(output_path),
        }

    existing_ids: set[str] = set()
    if Path(output_path).exists():
        existing_ids = {str(record.get("id")) for record in read_jsonl(output_path)}

    records = []
    errors: list[dict[str, str]] = []
    emitted = 0
    for record in read_jsonl(input_path):
        if str(record.get("id")) in existing_ids:
            continue
        try:
            records.append(
                prelabel_record(
                    record,
                    model_id="heuristic-local-dryrun" if not use_azure else os.getenv("AZURE_OPENAI_DEPLOYMENT_LABEL", os.getenv("AZURE_OPENAI_DEPLOYMENT", "azure-openai")),
                    provider="local_heuristic" if not use_azure else "azure_openai",
                    use_azure=use_azure,
                )
            )
        except (AzureOpenAIError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"id": str(record.get("id")), "error": str(exc)})
            if len(errors) >= 5:
                break
        emitted += 1
        if rate_limit_seconds:
            time.sleep(rate_limit_seconds)
        if limit is not None and emitted >= limit:
            break
    write_jsonl(output_path, records)
    status = "ok" if not errors else "partial_failed"
    return {"status": status, "created": len(records), "errors": errors, "output": str(output_path), "mock": mock, "provider": "local_heuristic" if not use_azure else "azure_openai"}
