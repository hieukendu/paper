from __future__ import annotations

"""QLoRA direct multi-label classifier for ViPragSent.

Unlike the archived JSON-generation SFT baseline, this keeps a lightweight
six-label classification head on top of the quantized chat backbone.  Model
selection and label thresholds use only the supplied development split.  The
``--checkpoint`` mode loads an already selected adapter/head and never trains.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


GRID = [round(value / 100, 2) for value in range(1, 100)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SYSTEM = (
    "Bạn là chuyên gia phân tích bình luận mạng xã hội tiếng Việt. "
    "Đọc chính xác hiện tượng ngữ dụng, kể cả tiếng lóng, emoji, mỉa mai và xen tiếng Anh."
)


def prompt(text: str) -> str:
    return (
        "Đánh giá độc lập sáu hiện tượng: hàm ý cảm xúc, châm biếm, mỉa mai, "
        "thành ngữ/nghĩa bóng, chuyển mã ngôn ngữ và chế giễu.\n\n"
        f"Bình luận: {text}"
    )


class ClassificationDataset(Dataset):
    def __init__(self, path: Path, *, with_labels: bool):
        self.rows = list(read_jsonl(path))
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        result = {"id": str(row["id"]), "text": row["text"]}
        if self.with_labels:
            result["labels"] = [float(row["labels"][name]) for name in PRAGMATIC_LABELS]
        return result


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, rows: list[dict]) -> dict:
        messages = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(row["text"])}]
            for row in rows
        ]
        rendered = [
            self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch = {**encoded, "ids": [row["id"] for row in rows]}
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        return batch


class QLoRAMultilabel(nn.Module):
    def __init__(self, backbone, hidden_size: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        # The quantized backbone is dispatched directly to CUDA.  Keep the
        # newly-created head with its bf16 hidden states instead of leaving it
        # on CPU (or silently casting every batch through fp32).
        device = next(backbone.parameters()).device
        self.head = nn.Linear(hidden_size, len(PRAGMATIC_LABELS), dtype=torch.bfloat16).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state
        last_index = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]
        # Inference can upcast quantized backbone activations to fp32 while
        # the trainable head remains bf16.  Cast explicitly so dev selection
        # follows the exact saved classifier rather than failing post-train.
        return self.head(self.dropout(pooled).to(self.head.weight.dtype))


def build_model(model_id: str, *, rank: int, alpha: int, dropout: float, attention: str) -> tuple[QLoRAMultilabel, object]:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModel.from_pretrained(
        model_id,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attention,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base = get_peft_model(base, lora)
    model = QLoRAMultilabel(base, int(base.config.hidden_size), dropout)
    return model, base


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.inference_mode()
def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], list[list[float]], list[list[int]] | None]:
    model.eval()
    ids, values, gold = [], [], []
    for batch in loader:
        batch = move(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        ids.extend(batch["ids"])
        values.extend(torch.sigmoid(logits).float().cpu().tolist())
        if "labels" in batch:
            gold.extend(batch["labels"].int().cpu().tolist())
    return ids, values, gold if gold else None


def fit_thresholds(values: list[list[float]], gold: list[list[int]]) -> tuple[dict[str, float], dict[str, float]]:
    thresholds, scores = {}, {}
    for column, label in enumerate(PRAGMATIC_LABELS):
        truth = [row[column] for row in gold]
        ranked = []
        for threshold in GRID:
            score = binary_macro_f1(truth, [int(row[column] >= threshold) for row in values])
            ranked.append((score, -abs(threshold - 0.5), -threshold, threshold))
        best = max(ranked)
        thresholds[label] = best[3]
        scores[label] = best[0] * 100
    return thresholds, scores


def score_fixed_threshold(values: list[list[float]], gold: list[list[int]], threshold: float) -> tuple[dict[str, float], dict[str, float]]:
    thresholds = {label: threshold for label in PRAGMATIC_LABELS}
    scores = {}
    for column, label in enumerate(PRAGMATIC_LABELS):
        scores[label] = binary_macro_f1(
            [row[column] for row in gold],
            [int(row[column] >= threshold) for row in values],
        ) * 100
    return thresholds, scores


def prediction_rows(ids: list[str], values: list[list[float]], thresholds: dict[str, float], system: str, seed: int) -> list[dict]:
    rows = []
    for record_id, scores in zip(ids, values, strict=True):
        probabilities_by_label = {label: float(scores[index]) for index, label in enumerate(PRAGMATIC_LABELS)}
        predictions = {label: int(probabilities_by_label[label] >= thresholds[label]) for label in PRAGMATIC_LABELS}
        rows.append(
            {
                "id": record_id,
                "system": system,
                "seed": seed,
                "predictions": predictions,
                "probabilities": probabilities_by_label,
            }
        )
    return rows


def save_checkpoint(directory: Path, model: QLoRAMultilabel, args: argparse.Namespace, thresholds: dict[str, float], scores: dict[str, float], epoch: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "adapter")
    torch.save(model.head.state_dict(), directory / "classification_head.pt")
    (directory / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": str(args.dev),
                "epoch": epoch,
                "thresholds": thresholds,
                "development_binary_macro_f1": scores,
                "development_macro_pragmatic_f1": sum(scores.values()) / len(scores),
                "architecture": {"pooling": "last_instruction_token", "classification_head": "linear_6"},
                "training_args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(model_id: str, directory: Path, args: argparse.Namespace) -> QLoRAMultilabel:
    from peft import PeftModel

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModel.from_pretrained(model_id, quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn_implementation)
    base.config.use_cache = False
    backbone = PeftModel.from_pretrained(base, directory / "adapter")
    model = QLoRAMultilabel(backbone, int(backbone.config.hidden_size), args.dropout)
    model.head.load_state_dict(torch.load(directory / "classification_head.pt", map_location="cpu", weights_only=True))
    return model


def run_prediction(model: QLoRAMultilabel, tokenizer, path: Path, output: Path, thresholds: dict[str, float], args: argparse.Namespace, *, with_labels: bool) -> tuple[dict[str, float] | None, int]:
    dataset = ClassificationDataset(path, with_labels=with_labels)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=Collator(tokenizer, args.max_length), num_workers=2, pin_memory=True)
    device = next(model.parameters()).device
    ids, values, gold = probabilities(model, loader, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, prediction_rows(ids, values, thresholds, args.system, args.seed))
    if gold is None:
        return None, len(ids)
    _, scores = fit_thresholds(values, gold)
    return scores, len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/vipragsent_test.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/flexible_optimization")
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Selected run directory for inference-only mode.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional deterministic cap on shuffled batches per epoch; recorded for resource-bounded experiments.",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--positive-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use one declared decision threshold for every label instead of fitting development thresholds.",
    )
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA multi-label training requires CUDA.")
    if args.positive_weight_power < 0:
        raise SystemExit("--positive-weight-power must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise SystemExit("--max-train-batches must be positive when supplied")
    if args.fixed_threshold is not None and not 0.0 < args.fixed_threshold < 1.0:
        raise SystemExit("--fixed-threshold must be strictly between 0 and 1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    started = time.monotonic()

    if args.checkpoint:
        selection = json.loads((args.checkpoint / "selection.json").read_text(encoding="utf-8"))
        model = load_checkpoint(args.model_id, args.checkpoint, args).to("cuda")
        scores, count = run_prediction(model, tokenizer, args.test, args.prediction_output, selection["thresholds"], args, with_labels=False)
        print(json.dumps({"status": "ok", "mode": "predict_only", "records": count, "predictions": str(args.prediction_output), "checkpoint": str(args.checkpoint), "elapsed_seconds": round(time.monotonic() - started, 3)}, indent=2))
        return 0

    train_data = ClassificationDataset(args.train, with_labels=True)
    dev_data = ClassificationDataset(args.dev, with_labels=True)
    collator = Collator(tokenizer, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, collate_fn=collator, num_workers=2, pin_memory=True, persistent_workers=True)
    model, _ = build_model(args.model_id, rank=args.rank, alpha=args.alpha, dropout=args.dropout, attention=args.attn_implementation)
    device = next(model.parameters()).device
    target = torch.tensor([row["labels"] for row in train_data], dtype=torch.float32)
    positive = target.sum(dim=0).clamp_min(1)
    pos_weight = ((len(train_data) - positive) / positive).pow(args.positive_weight_power).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, fused=True)
    train_batches_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps = max(1, math.ceil(train_batches_per_epoch / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    run_dir = args.output_root / args.system / str(args.seed)
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            if step > train_batches_per_epoch:
                break
            batch = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(batch["input_ids"], batch["attention_mask"]), batch["labels"], pos_weight=pos_weight) / args.grad_accum
            loss.backward(); loss_sum += float(loss.detach())
            if step % args.grad_accum == 0 or step == train_batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        _, dev_values, dev_gold = probabilities(model, dev_loader, device)
        thresholds, scores = (
            score_fixed_threshold(dev_values, dev_gold or [], args.fixed_threshold)
            if args.fixed_threshold is not None
            else fit_thresholds(dev_values, dev_gold or [])
        )
        macro = sum(scores.values()) / len(scores)
        history.append({"epoch": epoch, "train_loss": loss_sum, "train_batches": train_batches_per_epoch, "development_macro_pragmatic_f1": macro, "development_binary_macro_f1": scores, "thresholds": thresholds, "elapsed_seconds": round(time.monotonic() - started, 3)})
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if macro > best:
            best, stale = macro, 0
            save_checkpoint(run_dir, model, args, thresholds, scores, epoch)
        else:
            stale += 1
            if stale >= args.patience:
                break
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    # Reloading proves the saved selected adapter/head is usable before any test scoring.
    del model
    torch.cuda.empty_cache()
    selected = load_checkpoint(args.model_id, run_dir, args).to("cuda")
    _, count = run_prediction(selected, tokenizer, args.test, args.prediction_output, selection["thresholds"], args, with_labels=True)
    print(json.dumps({"status": "ok", "mode": "train_and_dev_score", "records": count, "predictions": str(args.prediction_output), "checkpoint": str(run_dir), "best_development_macro_pragmatic_f1": best, "selection": selection, "elapsed_seconds": round(time.monotonic() - started, 3)}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
