#!/usr/bin/env python3
"""Prediction-only zero/few-shot prompting with a frozen Vistral base model.

This runner intentionally never loads a ViPragSent QLoRA adapter.  It is a
deterministic inference-only candidate and records every raw completion and
parse failure for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vipragsent.data.schema import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.utils.io import read_jsonl, write_jsonl

SYSTEM_PROMPT = (
    "Bạn là bộ phân loại cảm xúc ngữ dụng tiếng Việt. Trả về duy nhất JSON có các trường "
    "implicit_sentiment, sarcasm, irony, idiom_figurative, code_switching, mocking (0 hoặc 1), "
    "polarity và emotion."
)


def compact_target(labels: dict) -> str:
    canonical = canonicalize_labels(labels)
    return json.dumps({**{name: int(canonical[name]) for name in PRAGMATIC_LABELS}, "polarity": canonical["polarity"], "emotion": canonical["emotion"]}, ensure_ascii=False, separators=(",", ":"))


def source(row: dict) -> str:
    return str((row.get("source") or {}).get("dataset") or "unknown")


def retrieve_examples(train: list[dict], query: list[dict], count: int) -> list[list[dict]]:
    if not count:
        return [[] for _ in query]
    train_groups: dict[str, list[int]] = {}
    query_groups: dict[str, list[int]] = {}
    for index, row in enumerate(train): train_groups.setdefault(source(row), []).append(index)
    for index, row in enumerate(query): query_groups.setdefault(source(row), []).append(index)
    result: list[list[dict]] = [[] for _ in query]
    for group, query_indices in query_groups.items():
        train_indices = train_groups.get(group, list(range(len(train))))
        transform = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, norm="l2")
        train_matrix = transform.fit_transform([str(train[index]["text"]) for index in train_indices])
        query_matrix = transform.transform([str(query[index]["text"]) for index in query_indices])
        similarity = (query_matrix @ train_matrix.T).toarray()
        top_count = min(count, len(train_indices))
        indices = np.argpartition(-similarity, kth=top_count - 1, axis=1)[:, :top_count]
        values = np.take_along_axis(similarity, indices, axis=1)
        order = np.argsort(-values, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        for position, query_index in enumerate(query_indices):
            result[query_index] = [train[train_indices[index]] for index in indices[position]]
    return result


def prompt(tokenizer, text: str, examples: list[dict]) -> str:
    demonstrations = "\n\n".join(f"Ví dụ {index + 1}:\nBình luận: {example['text']}\nJSON: {compact_target(example['labels'])}" for index, example in enumerate(examples))
    user = f"{demonstrations}\n\n" if demonstrations else ""
    user += f"Bình luận: {text}\nJSON:"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_labels(completion: str) -> dict:
    start, end = completion.find("{"), completion.rfind("}")
    raw = json.loads(completion[start : end + 1]) if start >= 0 and end >= start else {}
    labels = canonicalize_labels(raw.get("labels", raw))
    for name in PRAGMATIC_LABELS:
        labels[name] = int(labels.get(name) in (1, True, "1", "true", "True"))
    labels["polarity"] = labels["polarity"] if labels.get("polarity") in POLARITY_LABELS else "neutral"
    labels["emotion"] = labels["emotion"] if labels.get("emotion") in EMOTION_LABELS else "other"
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path); parser.add_argument("--few-shot-k", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--max-input-length", type=int, default=512); parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int); args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token; tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if args.few_shot_k and not args.train:
        raise SystemExit("--train is required when --few-shot-k is positive")
    rows = list(read_jsonl(args.data))[:args.limit]
    demonstrations = retrieve_examples(list(read_jsonl(args.train)), rows, args.few_shot_k) if args.few_shot_k else [[] for _ in rows]
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer([prompt(tokenizer, str(row["text"]), demonstrations[start + offset]) for offset, row in enumerate(batch)], return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length).to(model.device)
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            width = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for row, completion in zip(batch, completions, strict=True):
                try:
                    labels = parse_labels(completion); parse_error = None
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    labels = {**{name: 0 for name in PRAGMATIC_LABELS}, "polarity": "neutral", "emotion": "other"}; parse_error = str(exc)
                output.append({"id": row["id"], "system": "frozen_vistral_base_few_shot", "predictions": labels, "generation": completion, "parse_error": parse_error, "few_shot_k": args.few_shot_k, "frozen_weight_compliance": {"adapter_loaded": False, "neural_weight_updates": False, "optimizer_or_backward_called": False}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    print(json.dumps({"records": len(output), "parse_errors": sum(row["parse_error"] is not None for row in output), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
