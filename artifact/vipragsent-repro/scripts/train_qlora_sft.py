from __future__ import annotations

"""Memory-bounded QLoRA baseline for the downloaded Vietnamese 7B models.

This is deliberately a classification SFT baseline: it emits the unified
ViPragSent label JSON directly and does not claim to implement the encoder
rationale auxiliary.  It is compatible with a 20 GB H100 MIG slice through
NF4, gradient checkpointing and a micro-batch of one.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from vipragsent.data.schema import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, canonicalize_labels
from vipragsent.utils.io import read_jsonl


POLARITIES = ("positive", "neutral", "negative")
EMOTIONS = ("enjoyment", "sadness", "anger", "disgust", "fear", "surprise", "other")
SYSTEM_PROMPT = (
    "Bạn là bộ phân loại cảm xúc ngữ dụng tiếng Việt. Trả về duy nhất JSON có các trường "
    "implicit_sentiment, sarcasm, irony, idiom_figurative, code_switching, mocking (0 hoặc 1), "
    "polarity và emotion."
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nBình luận: {text}\nJSON:"


def target(labels: dict) -> str:
    canonical = canonicalize_labels(labels)
    return json.dumps(
        {**{name: int(canonical[name]) for name in PRAGMATIC_LABELS}, "polarity": canonical["polarity"], "emotion": canonical["emotion"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


class SftDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {"id": row["id"], "prompt": prompt(row["text"]), "target": target(row["labels"])}


class TrainCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        inputs, labels = [], []
        for row in rows:
            prefix = self.tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
            suffix = self.tokenizer(row["target"] + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (prefix + suffix)[-self.max_length :]
            prefix_len = max(0, len(ids) - len(suffix))
            inputs.append(ids)
            labels.append([-100] * prefix_len + ids[prefix_len:])
        max_len = max(len(item) for item in inputs)
        pad = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([item + [pad] * (max_len - len(item)) for item in inputs]),
            "attention_mask": torch.tensor([[1] * len(item) + [0] * (max_len - len(item)) for item in inputs]),
            "labels": torch.tensor([item + [-100] * (max_len - len(item)) for item in labels]),
        }


def load_model(model_id: str, attn_implementation: str):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def parse_labels(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    raw = json.loads(text[start : end + 1]) if start >= 0 and end >= start else {}
    labels = canonicalize_labels(raw.get("labels", raw))
    for name in PRAGMATIC_LABELS:
        labels[name] = int(labels.get(name) in (1, True, "1", "true", "True"))
    labels["polarity"] = labels["polarity"] if labels.get("polarity") in POLARITY_LABELS else "neutral"
    labels["emotion"] = labels["emotion"] if labels.get("emotion") in EMOTION_LABELS else "other"
    return labels


@torch.inference_mode()
def predict(
    model,
    tokenizer,
    path: Path,
    output: Path,
    system: str,
    seed: int,
    max_length: int,
    batch_size: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    count = 0
    rows = list(read_jsonl(path))
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    with output.open("w", encoding="utf-8") as handle:
        index = 0
        current_batch_size = batch_size
        while index < len(rows):
            batch = rows[index : index + current_batch_size]
            encoded = tokenizer(
                [prompt(row["text"]) for row in batch],
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                truncation=True,
                max_length=max_length,
            ).to(device)
            try:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=96,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except torch.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
                print(f"Prediction OOM; retrying with batch size {current_batch_size}.", flush=True)
                continue
            prompt_width = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for row, completion in zip(batch, completions, strict=True):
                try:
                    labels = parse_labels(completion)
                    parse_error = None
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    labels = {**{name: 0 for name in PRAGMATIC_LABELS}, "polarity": "neutral", "emotion": "other"}
                    parse_error = str(exc)
                handle.write(json.dumps({"id": row["id"], "system": system, "seed": seed, "predictions": labels, "generation": completion, "parse_error": parse_error}, ensure_ascii=False) + "\n")
                count += 1
            index += len(batch)
    tokenizer.padding_side = original_padding_side
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/vipragsent_test.jsonl")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--prediction-batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--prediction-root", type=Path, default=ROOT / "results/predictions/main_pragmatic")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires CUDA.")
    if args.prediction_batch_size < 1:
        raise SystemExit("--prediction-batch-size must be positive.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = load_model(args.model_id, args.attn_implementation)
    device = next(model.parameters()).device
    train = DataLoader(
        SftDataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainCollator(tokenizer, args.max_length),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, fused=True)
    steps = max(1, (len(train) + args.grad_accum - 1) // args.grad_accum * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.03), steps)
    run_dir = args.output_root / args.system / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach())
            if step % args.grad_accum == 0 or step == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": running, "elapsed_seconds": round(time.monotonic() - started, 3)})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    model.save_pretrained(run_dir / "adapter")
    prediction = args.prediction_root / args.system / f"{args.seed}.jsonl"
    prediction_count = predict(
        model,
        tokenizer,
        args.test,
        prediction,
        args.system,
        args.seed,
        args.max_length,
        args.prediction_batch_size,
    )
    manifest = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "seed": args.seed,
        "model_id": args.model_id,
        "device": str(device),
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "adapter": str(run_dir / "adapter"),
        "predictions": str(prediction),
        "prediction_records": prediction_count,
        "training_type": "nf4_qlora_sft",
        "performance_settings": {
            "attention_implementation": args.attn_implementation,
            "optimizer": "fused_adamw_trainable_lora_only",
            "prediction_batch_size": args.prediction_batch_size,
            "generation_kv_cache": True,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
