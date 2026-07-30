from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The school image ships an ONNX build generated against an older protobuf.
# Transformers imports torchvision/ONNX while loading text encoders, although
# this experiment never uses ONNX.  This compatibility fallback affects only
# that import path and keeps model training on the compiled PyTorch backend.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl

POLARITIES = ["negative", "neutral", "positive"]
EMOTIONS = ["enjoyment", "sadness", "anger", "disgust", "fear", "surprise", "other"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonlDataset(Dataset):
    def __init__(self, path: Path, rationales: dict[str, str] | None = None):
        self.records = list(read_jsonl(path))
        self.rationales = rationales or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        labels = record["labels"]
        return {
            "id": record["id"],
            "text": record["text"],
            "pragmatic": [float(labels[name]) for name in PRAGMATIC_LABELS],
            "polarity": POLARITIES.index(labels["polarity"]),
            "emotion": EMOTIONS.index(labels["emotion"]),
            "rationale": self.rationales.get(record["id"], ""),
        }


class JsonlPredictionDataset(Dataset):
    """Dataset used for inference over records whose labels may be incomplete."""

    def __init__(self, path: Path):
        self.records = list(read_jsonl(path))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {"id": record["id"], "text": record["text"]}


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        batch = {**encoded, "ids": [row["id"] for row in rows]}
        if "pragmatic" not in rows[0]:
            return batch
        rationales = self.tokenizer(
            [row["rationale"] or row["text"] for row in rows], padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        batch.update(
            {
                "rationale_input_ids": rationales["input_ids"],
                "rationale_attention_mask": rationales["attention_mask"],
                "pragmatic": torch.tensor([row["pragmatic"] for row in rows]),
                "polarity": torch.tensor([row["polarity"] for row in rows]),
                "emotion": torch.tensor([row["emotion"] for row in rows]),
            }
        )
        return batch


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_id: str,
        *,
        uncertainty: bool,
        rationale_aux: bool,
        encoder=None,
        pooling: str = "cls",
        dropout: float = 0.1,
        pragmatic_head: str = "linear",
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        feature_dim = hidden * {"cls": 1, "mean": 1, "max": 1, "clsmean": 2, "meanmax": 2, "clsmeanmax": 3}[pooling]
        self.dropout = nn.Dropout(dropout)
        if pragmatic_head == "linear":
            self.pragmatic = nn.Linear(feature_dim, len(PRAGMATIC_LABELS))
        elif pragmatic_head == "mlp":
            self.pragmatic = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(PRAGMATIC_LABELS)),
            )
        elif pragmatic_head == "residual_mlp":
            self.pragmatic = ResidualPragmaticHead(feature_dim, hidden, dropout)
        else:
            raise ValueError(f"unsupported pragmatic head: {pragmatic_head}")
        self.polarity = nn.Linear(feature_dim, len(POLARITIES))
        self.emotion = nn.Linear(feature_dim, len(EMOTIONS))
        self.rationale_projection = nn.Linear(feature_dim, hidden) if rationale_aux else None
        self.log_vars = nn.Parameter(torch.zeros(4)) if uncertainty else None

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        elif self.pooling in {"max", "clsmean", "meanmax", "clsmeanmax"}:
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            maximum = hidden.masked_fill(~attention_mask.unsqueeze(-1).bool(), torch.finfo(hidden.dtype).min).max(dim=1).values
            if self.pooling == "max":
                pooled = maximum
            elif self.pooling == "clsmean":
                pooled = torch.cat([hidden[:, 0], mean], dim=-1)
            elif self.pooling == "meanmax":
                pooled = torch.cat([mean, maximum], dim=-1)
            else:
                pooled = torch.cat([hidden[:, 0], mean, maximum], dim=-1)
        else:
            raise ValueError(f"unsupported pooling method: {self.pooling}")
        pooled = self.dropout(pooled)
        return self.pragmatic(pooled), self.polarity(pooled), self.emotion(pooled), pooled


class ResidualPragmaticHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.project = nn.Linear(feature_dim, hidden)
        self.block = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(hidden, len(PRAGMATIC_LABELS))

    def forward(self, pooled):
        base = self.project(pooled)
        return self.output(base + self.block(base))


def load_rationales(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return {str(row["id"]): str(row.get("rationale") or "") for row in read_jsonl(path)}


def task_loss(
    model,
    batch,
    device,
    *,
    beta,
    auxiliary_weight,
    focal_gamma,
    pragmatic_label_weights,
    use_emotion,
    use_polarity,
    use_rationale,
    pragmatic_pos_weight=None,
):
    pragmatic, polarity, emotion, pooled = model(batch["input_ids"], batch["attention_mask"])
    pragmatic_loss = F.binary_cross_entropy_with_logits(
        pragmatic,
        batch["pragmatic"],
        pos_weight=pragmatic_pos_weight,
        reduction="none",
    )
    if focal_gamma:
        probabilities = torch.sigmoid(pragmatic)
        p_t = probabilities * batch["pragmatic"] + (1 - probabilities) * (1 - batch["pragmatic"])
        pragmatic_loss = ((1 - p_t).pow(focal_gamma) * pragmatic_loss)
    if pragmatic_label_weights is None:
        pragmatic_loss = pragmatic_loss.mean()
    else:
        label_weights = pragmatic_label_weights.to(pragmatic_loss.device, pragmatic_loss.dtype)
        pragmatic_loss = (
            (pragmatic_loss * label_weights).sum(dim=-1) / label_weights.sum().clamp_min(1e-8)
        ).mean()
    # Keep task identity separate from its contribution.  In particular, a
    # disabled task must not add its uncertainty regularizer (``log_var``).
    # Adding ``log_var`` for a zero loss drives it to -infinity and changes the
    # relative weighting of the active objectives.
    task_terms = [(0, pragmatic_loss)]
    if use_polarity:
        task_terms.append((1, auxiliary_weight * F.cross_entropy(polarity, batch["polarity"])))
    if use_emotion:
        task_terms.append((2, auxiliary_weight * F.cross_entropy(emotion, batch["emotion"])))
    if use_rationale:
        with torch.no_grad():
            embeddings = model.encoder.get_input_embeddings()(batch["rationale_input_ids"])
            mask = batch["rationale_attention_mask"].unsqueeze(-1)
            target = (embeddings * mask).sum(1) / mask.sum(1).clamp_min(1)
        task_terms.append((3, beta * F.mse_loss(model.rationale_projection(pooled), target)))
    if model.log_vars is None:
        total = sum(loss for _, loss in task_terms)
    else:
        total = sum(
            torch.exp(-model.log_vars[index]) * loss + model.log_vars[index]
            for index, loss in task_terms
        )
    losses_by_task = [0.0] * 4
    for index, loss in task_terms:
        losses_by_task[index] = loss.detach().item()
    return total, (pragmatic, polarity, emotion), losses_by_task


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_local_encoder(model_id: str, state_dict_path: Path | None):
    """Load a trusted local base encoder without relying on legacy pickle loading.

    Older model archives use ``pytorch_model.bin`` and recent Transformers
    disallows loading those through ``from_pretrained`` on this Torch build.
    Explicit ``weights_only=True`` keeps both fine-tuning and inference on the
    same audited local tensor values.
    """
    if state_dict_path is None:
        return None
    if not state_dict_path.is_file():
        raise SystemExit(f"missing local base-model state dict: {state_dict_path}")
    config = AutoConfig.from_pretrained(model_id)
    encoder = AutoModel.from_config(config)
    base_state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    encoder_state = {
        key.removeprefix("roberta."): value
        for key, value in base_state.items()
        if key.startswith("roberta.") and key != "roberta.embeddings.position_ids"
    }
    if not encoder_state:
        encoder_state = base_state
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    # The old RoBERTa archives include one non-parameter position-id buffer.
    # Any actual encoder tensor mismatch would invalidate a training run.
    # ViSoBERT's archive deliberately omits the unused RoBERTa pooler; this
    # classifier consumes ``last_hidden_state`` directly, so initializing that
    # never-called module is harmless.  Do not permit any missing encoder layer.
    missing = [
        key for key in incompatible.missing_keys
        if key != "embeddings.position_ids" and not key.startswith("pooler.")
    ]
    if missing or incompatible.unexpected_keys:
        raise SystemExit(
            "local base-model state dict does not match encoder: "
            f"missing={missing[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
        )
    return encoder


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); gold = []; pred = []
    for batch in loader:
        batch = move(batch, device)
        logits, _, _, _ = model(batch["input_ids"], batch["attention_mask"])
        gold.extend(batch["pragmatic"].cpu().int().tolist())
        pred.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
    values = []
    for i in range(len(PRAGMATIC_LABELS)):
        values.append(binary_macro_f1([row[i] for row in gold], [row[i] for row in pred]))
    return sum(values) / len(values), dict(zip(PRAGMATIC_LABELS, values))


@torch.no_grad()
def collect_pragmatic_probabilities(model, loader, device):
    """Return development IDs, gold labels, and unthresholded probabilities."""
    model.eval(); ids = []; gold = []; probabilities = []
    for batch in loader:
        batch = move(batch, device)
        logits, _, _, _ = model(batch["input_ids"], batch["attention_mask"])
        ids.extend(batch["ids"])
        gold.extend(batch["pragmatic"].cpu().int().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return ids, gold, probabilities


def score_thresholds(gold, probabilities, thresholds):
    values = []
    for label_index in range(len(PRAGMATIC_LABELS)):
        values.append(binary_macro_f1(
            [row[label_index] for row in gold],
            [int(row[label_index] >= thresholds[label_index]) for row in probabilities],
        ))
    return sum(values) / len(values), dict(zip(PRAGMATIC_LABELS, values))


def calibrate_thresholds(gold, probabilities, *, step: float):
    """Fit independent label thresholds on a calibration partition only."""
    if not 0 < step <= 0.5:
        raise ValueError("threshold step must be in (0, 0.5]")
    candidates = [round(value * step, 6) for value in range(1, int(1 / step))]
    thresholds = []
    for label_index in range(len(PRAGMATIC_LABELS)):
        label_gold = [row[label_index] for row in gold]
        label_probabilities = [row[label_index] for row in probabilities]
        best_threshold, best_score = 0.5, -1.0
        for threshold in candidates:
            score = binary_macro_f1(label_gold, [int(value >= threshold) for value in label_probabilities])
            # Deterministic tie-breaker: prefer a threshold closest to 0.5.
            if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_threshold, best_score = threshold, score
        thresholds.append(best_threshold)
    return thresholds


def calibration_selection_partition(ids):
    """Stable, ID-hashed 50/50 dev split for threshold-aware selection."""
    calibration, selection = [], []
    for index, record_id in enumerate(ids):
        bucket = hashlib.sha256(str(record_id).encode("utf-8")).digest()[0] & 1
        (calibration if bucket == 0 else selection).append(index)
    if not calibration or not selection:
        raise ValueError("development calibration/selection partition is empty")
    return calibration, selection


def threshold_aware_evaluate(model, loader, device, *, step: float):
    ids, gold, probabilities = collect_pragmatic_probabilities(model, loader, device)
    calibration, selection = calibration_selection_partition(ids)
    calibration_gold = [gold[index] for index in calibration]
    calibration_probabilities = [probabilities[index] for index in calibration]
    thresholds = calibrate_thresholds(calibration_gold, calibration_probabilities, step=step)
    selection_gold = [gold[index] for index in selection]
    selection_probabilities = [probabilities[index] for index in selection]
    macro, per_label = score_thresholds(selection_gold, selection_probabilities, thresholds)
    return macro, per_label, dict(zip(PRAGMATIC_LABELS, thresholds)), {
        "calibration_records": len(calibration),
        "selection_records": len(selection),
        "partition": "sha256(id) parity",
    }


@torch.no_grad()
def write_predictions(model, loader, device, output: Path, system: str, seed: int):
    model.eval(); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for batch in loader:
            batch = move(batch, device)
            prag, pol, emo, _ = model(batch["input_ids"], batch["attention_mask"])
            prag_prob = torch.sigmoid(prag).cpu(); pol_prob = torch.softmax(pol, -1).cpu(); emo_prob = torch.softmax(emo, -1).cpu()
            for i, record_id in enumerate(batch["ids"]):
                predictions = {name: int(prag_prob[i, j] >= 0.5) for j, name in enumerate(PRAGMATIC_LABELS)}
                predictions["polarity"] = POLARITIES[int(pol_prob[i].argmax())]
                predictions["emotion"] = EMOTIONS[int(emo_prob[i].argmax())]
                probabilities = {name: float(prag_prob[i, j]) for j, name in enumerate(PRAGMATIC_LABELS)}
                probabilities["polarity"] = {name: float(pol_prob[i, j]) for j, name in enumerate(POLARITIES)}
                sarcasm = float(prag_prob[i, PRAGMATIC_LABELS.index("sarcasm")])
                irony = float(prag_prob[i, PRAGMATIC_LABELS.index("irony")])
                mocking = float(prag_prob[i, PRAGMATIC_LABELS.index("mocking")])
                raw_pragmatic_polarity = {
                    "positive": float(pol_prob[i, POLARITIES.index("positive")]) * (1.0 - irony),
                    "negative": float(pol_prob[i, POLARITIES.index("negative")]) * (1.0 - sarcasm),
                    "neutral": float(pol_prob[i, POLARITIES.index("neutral")]),
                    "ironic-positive": irony * float(pol_prob[i, POLARITIES.index("positive")]),
                    "sarcastic-negative": sarcasm * float(pol_prob[i, POLARITIES.index("negative")]),
                    "mocking": mocking,
                }
                norm = sum(raw_pragmatic_polarity.values()) or 1.0
                probabilities["pragmatic_polarity"] = {key: value / norm for key, value in raw_pragmatic_polarity.items()}
                row = {
                    "id": record_id, "system": system, "seed": seed,
                    "predictions": predictions, "probabilities": probabilities,
                    "logits": {
                        "pragmatic": [float(value) for value in prag[i].detach().cpu()],
                        "polarity": [float(value) for value in pol[i].detach().cpu()],
                        "emotion": [float(value) for value in emo[i].detach().cpu()],
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/vipragsent_test.jsonl")
    parser.add_argument("--rationales", type=Path, default=ROOT / "data/generated/rationales.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--prediction-root", type=Path, default=ROOT / "results/predictions/main_pragmatic")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--pooling", choices=("cls", "mean", "max", "clsmean", "meanmax", "clsmeanmax"), default="cls")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pragmatic-head", choices=("linear", "mlp", "residual_mlp"), default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--auxiliary-weight",
        type=float,
        default=1.0,
        help="Relative weight for the polarity and emotion auxiliary losses.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss gamma for pragmatic labels; 0 retains BCE.",
    )
    parser.add_argument(
        "--positive-weight-power",
        type=float,
        default=0.0,
        help="Use ((negative / positive) ** power) for each pragmatic BCE label; 0 disables class weighting.",
    )
    parser.add_argument(
        "--positive-weight-powers",
        help=(
            "Optional comma-separated per-label positive-weight powers in PRAGMATIC_LABELS order. "
            "Overrides --positive-weight-power; use 0 for a label with no upweighting."
        ),
    )
    parser.add_argument(
        "--pragmatic-label-weights",
        help=(
            "Comma-separated non-negative loss weights in PRAGMATIC_LABELS order. "
            "Use a one-hot vector for a label-specific expert."
        ),
    )
    parser.add_argument(
        "--selection-label",
        choices=PRAGMATIC_LABELS,
        help="Select the checkpoint by this label's dev macro-F1 instead of six-label macro-F1.",
    )
    parser.add_argument(
        "--threshold-aware-selection",
        action="store_true",
        help="Select epochs with per-label thresholds calibrated on a deterministic dev calibration split.",
    )
    parser.add_argument(
        "--threshold-grid-step",
        type=float,
        default=0.01,
        help="Threshold grid step for dev-only calibrated checkpoint selection.",
    )
    parser.add_argument("--no-rationale", action="store_true")
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument("--no-polarity", action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Existing checkpoint for prediction-only mode.")
    parser.add_argument("--predict-data", type=Path, help="JSONL data to score without training.")
    parser.add_argument("--prediction-output", type=Path, help="Explicit prediction JSONL output path.")
    parser.add_argument(
        "--skip-test-prediction",
        action="store_true",
        help="Do not write a prediction file after training; useful for development-only ablations.",
    )
    parser.add_argument(
        "--local-base-pytorch-bin",
        type=Path,
        help=(
            "Trusted local base-model state dict for prediction-only compatibility when "
            "the installed Transformers version rejects an older Torch runtime. "
            "Loaded with weights_only=True; this never updates model parameters."
        ),
    )
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.predict_data):
        raise SystemExit("--checkpoint and --predict-data must be supplied together for prediction-only mode.")
    seed_everything(args.seed)
    pragmatic_label_weights = None
    if args.pragmatic_label_weights:
        values = [float(value.strip()) for value in args.pragmatic_label_weights.split(",")]
        if len(values) != len(PRAGMATIC_LABELS) or any(value < 0 for value in values) or not any(values):
            raise SystemExit(
                "--pragmatic-label-weights must give six non-negative values with at least one positive weight"
            )
        pragmatic_label_weights = torch.tensor(values, dtype=torch.float32)
    if args.positive_weight_power < 0:
        raise SystemExit("--positive-weight-power must be non-negative")
    positive_weight_powers = [args.positive_weight_power] * len(PRAGMATIC_LABELS)
    if args.positive_weight_powers:
        positive_weight_powers = [float(value.strip()) for value in args.positive_weight_powers.split(",")]
        if len(positive_weight_powers) != len(PRAGMATIC_LABELS) or any(value < 0 for value in positive_weight_powers):
            raise SystemExit("--positive-weight-powers must give six non-negative values")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    rationales = load_rationales(args.rationales)
    if args.checkpoint:
        collate = Collator(tokenizer, args.max_length)
        prediction_data = DataLoader(
            JsonlPredictionDataset(args.predict_data),
            batch_size=args.batch_size * 2,
            collate_fn=collate,
            num_workers=2,
        )
        encoder = load_local_encoder(args.model_id, args.local_base_pytorch_bin)
        model = MultiTaskModel(
            args.model_id,
            uncertainty=not args.no_uncertainty,
            rationale_aux=not args.no_rationale,
            encoder=encoder,
            pooling=args.pooling,
            dropout=args.dropout,
            pragmatic_head=args.pragmatic_head,
        ).to(device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        output = args.prediction_output or args.prediction_root / args.system / f"{args.seed}.jsonl"
        write_predictions(model, prediction_data, device, output, args.system, args.seed)
        print(json.dumps({"status": "ok", "mode": "predict_only", "checkpoint": str(args.checkpoint), "records": len(prediction_data.dataset), "predictions": str(output)}, indent=2))
        return 0
    train_ids = {str(row["id"]) for row in read_jsonl(args.train)}
    if not args.no_rationale and not train_ids.issubset(rationales):
        raise SystemExit(
            f"Rationale coverage incomplete: {len(train_ids & set(rationales))}/{len(train_ids)}. "
            "Resume scripts/06_generate_rationales.py or pass --no-rationale for the ablation."
        )
    collate = Collator(tokenizer, args.max_length)
    train = DataLoader(JsonlDataset(args.train, rationales), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    dev = DataLoader(JsonlDataset(args.dev), batch_size=args.batch_size * 2, collate_fn=collate, num_workers=2)
    test = DataLoader(JsonlDataset(args.test), batch_size=args.batch_size * 2, collate_fn=collate, num_workers=2)
    encoder = load_local_encoder(args.model_id, args.local_base_pytorch_bin)
    model = MultiTaskModel(
        args.model_id,
        uncertainty=not args.no_uncertainty,
        rationale_aux=not args.no_rationale,
        encoder=encoder,
        pooling=args.pooling,
        dropout=args.dropout,
        pragmatic_head=args.pragmatic_head,
    ).to(device)
    if args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1 or args.gradient_clip <= 0:
        raise SystemExit("weight decay must be non-negative, warmup in [0,1), and gradient clip positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.bf16)
    run_dir = args.output_root / args.system / str(args.seed); run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best_selection = -1.0; best_macro = -1.0; best_thresholds = None; selection_protocol = None; stale = 0; history = []
    pragmatic_pos_weight = None
    if any(positive_weight_powers):
        pragmatic_targets = torch.tensor(
            [row["pragmatic"] for row in train.dataset], dtype=torch.float32
        )
        positive = pragmatic_targets.sum(dim=0).clamp_min(1)
        negative = len(train.dataset) - positive
        powers = torch.tensor(positive_weight_powers, dtype=torch.float32)
        pragmatic_pos_weight = (negative / positive).pow(powers).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        for step, batch in enumerate(train, 1):
            batch = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=device.type == "cuda"):
                loss, _, _ = task_loss(
                    model,
                    batch,
                    device,
                    beta=args.beta,
                    auxiliary_weight=args.auxiliary_weight,
                    focal_gamma=args.focal_gamma,
                    pragmatic_label_weights=pragmatic_label_weights,
                    use_emotion=not args.no_emotion,
                    use_polarity=not args.no_polarity,
                    use_rationale=not args.no_rationale,
                    pragmatic_pos_weight=pragmatic_pos_weight,
                )
                loss = loss / args.grad_accum
            scaler.scale(loss).backward(); running += loss.item()
            if step % args.grad_accum == 0 or step == len(train):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        if args.threshold_aware_selection:
            score, per_label, calibrated_thresholds, selection_protocol = threshold_aware_evaluate(
                model, dev, device, step=args.threshold_grid_step,
            )
        else:
            score, per_label = evaluate(model, dev, device)
            calibrated_thresholds = None
        selection_score = per_label[args.selection_label] if args.selection_label else score
        history.append({"epoch": epoch, "train_loss": running, "dev_macro_pragmatic_f1": score, "selection_score": selection_score, "per_label": per_label, "thresholds": calibrated_thresholds, "elapsed_seconds": round(time.monotonic() - started, 3)})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if selection_score > best_selection:
            best_selection = selection_score; best_macro = score; stale = 0
            best_thresholds = calibrated_thresholds
            torch.save({"model": model.state_dict(), "args": vars(args), "best_dev": best_selection, "selection_thresholds": best_thresholds}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    prediction = None
    if not args.skip_test_prediction:
        prediction = args.prediction_root / args.system / f"{args.seed}.jsonl"
        write_predictions(model, test, device, prediction, args.system, args.seed)
    manifest = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "seed": args.seed,
        "model_id": args.model_id,
        "device": str(device),
        "bf16": bool(args.bf16),
        "train_records": len(train.dataset),
        "dev_records": len(dev.dataset),
        "test_records": len(test.dataset),
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "best_dev_macro_pragmatic_f1": best_macro,
        "best_dev_selection_score": best_selection,
        "checkpoint": str(run_dir / "best.pt"),
        "predictions": str(prediction) if prediction is not None else None,
        "rationales_loaded": len(rationales),
        "flags": {"no_rationale": args.no_rationale, "no_emotion": args.no_emotion, "no_polarity": args.no_polarity, "no_uncertainty": args.no_uncertainty},
        "advanced_options": {
            "pooling": args.pooling,
            "dropout": args.dropout,
            "pragmatic_head": args.pragmatic_head,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_clip": args.gradient_clip,
            "auxiliary_weight": args.auxiliary_weight,
            "focal_gamma": args.focal_gamma,
            "positive_weight_power": args.positive_weight_power,
            "positive_weight_powers": positive_weight_powers,
            "pragmatic_pos_weight": pragmatic_pos_weight.cpu().tolist() if pragmatic_pos_weight is not None else None,
            "pragmatic_label_weights": pragmatic_label_weights.tolist() if pragmatic_label_weights is not None else None,
            "selection_label": args.selection_label,
            "threshold_aware_selection": args.threshold_aware_selection,
            "threshold_grid_step": args.threshold_grid_step,
            "best_selection_thresholds": best_thresholds,
            "selection_protocol": selection_protocol,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
