#!/usr/bin/env python3
"""Frozen causal-LM embedding k-NN for ViPragSent.

This uses only the supplied pretrained causal-LM base in evaluation mode.  It
does not load a task adapter, generate labels, or update model weights; train
labels are used solely by the non-parametric k-NN retrieval rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_embedding_knn import ALPHAS, K_VALUES, THRESHOLDS, neighbors
from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


class TextDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def load_encoder(base: Path):
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def embed(path: Path, *, tokenizer, model, max_length: int, batch_size: int, pooling: str) -> tuple[list[dict], torch.Tensor]:
    dataset = TextDataset(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def collate(rows: list[dict]) -> dict:
        tokens = tokenizer([str(row["text"]) for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {"rows": rows, **tokens}

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=2)
    vectors, rows_out = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            hidden_states = model(**inputs).last_hidden_state
            if pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                hidden = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                positions = inputs["attention_mask"].sum(dim=1).sub(1)
                hidden = hidden_states[torch.arange(len(positions), device=device), positions]
            vectors.append(F.normalize(hidden.float(), dim=-1).cpu())
            rows_out.extend(batch["rows"])
    return rows_out, torch.cat(vectors)


def base_rows(path: Path) -> list[dict]:
    return _ensemble_rows(sorted(path.glob("reproduction_*.jsonl")))


def fit(args: argparse.Namespace) -> None:
    tokenizer, model = load_encoder(args.base)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    dev_rows, dev_embeddings = embed(args.dev_gold, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    retrieval = neighbors(train_embeddings, dev_embeddings, train_labels)
    base = {row["id"]: row for row in base_rows(args.dev_predictions)}
    selected = {}
    for label_index, label in enumerate(PRAGMATIC_LABELS):
        truth = [int(row["labels"][label]) for row in dev_rows]
        model_scores = torch.tensor([float(base[row["id"]]["probabilities"][label]) for row in dev_rows])
        best = (-1.0, None, None, None)
        for k, scores in retrieval.items():
            retrieval_scores = scores[:, label_index]
            for alpha in ALPHAS:
                blend = alpha * retrieval_scores + (1.0 - alpha) * model_scores
                for threshold in THRESHOLDS:
                    value = binary_macro_f1(truth, [int(score >= threshold) for score in blend.tolist()]) * 100
                    candidate = (value, -abs(alpha - 0.5), -abs(threshold - 0.5), k, alpha, threshold)
                    if candidate > (best[0], -abs(best[2] - 0.5) if best[2] is not None else -9, -abs(best[3] - 0.5) if best[3] is not None else -9, best[1] or 0, best[2] or 0, best[3] or 0):
                        best = (value, k, alpha, threshold)
        selected[label] = {"k": best[1], "alpha_knn": best[2], "threshold": best[3], "development_binary_macro_f1": round(best[0], 4)}
    payload = {
        "method": "frozen_pretrained_causal_lm_embedding_knn_blended_with_frozen_vipragsent_head",
        "selection_split": "development", "base": str(args.base), "adapter_loaded": False,
        "max_length": args.max_length, "pooling": args.pooling, "k_values": K_VALUES, "alpha_values": ALPHAS, "labels": selected,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: rule["development_binary_macro_f1"] for label, rule in selected.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer, model = load_encoder(args.base)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    data_rows, data_embeddings = embed(args.data, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    needed = sorted({rule["k"] for rule in config["labels"].values()})
    from frozen_embedding_knn import K_VALUES as imported_k_values
    import frozen_embedding_knn
    previous = imported_k_values[:]
    frozen_embedding_knn.K_VALUES = needed
    retrieval = neighbors(train_embeddings, data_embeddings, train_labels)
    frozen_embedding_knn.K_VALUES = previous
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    position = {row["id"]: index for index, row in enumerate(data_rows)}
    for row in rows:
        for label_index, label in enumerate(PRAGMATIC_LABELS):
            rule = config["labels"][label]
            value = rule["alpha_knn"] * float(retrieval[rule["k"]][position[row["id"]], label_index]) + (1 - rule["alpha_knn"]) * float(row["probabilities"][label])
            row["probabilities"][f"{label}_causal_knn_blend"] = value
            row["predictions"][label] = int(value >= rule["threshold"])
        row["system"] = "vipragsent_frozen_causal_embedding_knn"
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        command = commands.add_parser(name)
        command.add_argument("--base", type=Path, required=True); command.add_argument("--train", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=32)
        if name == "fit":
            command.add_argument("--dev-gold", type=Path, required=True); command.add_argument("--dev-predictions", type=Path, required=True)
            command.add_argument("--max-length", type=int, default=128); command.add_argument("--pooling", choices=["last", "mean"], default="mean")
            command.add_argument("--output", type=Path, required=True); command.set_defaults(func=fit)
        else:
            command.add_argument("--config", type=Path, required=True); command.add_argument("--data", type=Path, required=True)
            command.add_argument("--predictions", type=Path, required=True); command.add_argument("--base-config", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True); command.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
