"""Train-only character TF-IDF code-switching experts for development selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-min", type=int, required=True)
    parser.add_argument("--ngram-max", type=int, required=True)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    train, dev = list(read_jsonl(args.train)), list(read_jsonl(args.dev))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(args.ngram_min, args.ngram_max), min_df=2, sublinear_tf=True, max_features=150000)
    x_train = vectorizer.fit_transform([row["text"] for row in train]); x_dev = vectorizer.transform([row["text"] for row in dev])
    y = [int(row["labels"]["code_switching"]) for row in train]
    weights = [1.0 if value else args.negative_weight for value in y]
    model = LogisticRegression(C=args.c, max_iter=1000, solver="liblinear", random_state=20261001)
    model.fit(x_train, y, sample_weight=weights)
    probabilities = model.predict_proba(x_dev)[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row, probability in zip(dev, probabilities):
            handle.write(json.dumps({"id": row["id"], "system": "code_switch_char_expert", "probabilities": {"code_switching": float(probability)}}) + "\n")
    (args.output.with_suffix(".metadata.json")).write_text(json.dumps({"train_records": len(train), "dev_records": len(dev), "ngram_range": [args.ngram_min, args.ngram_max], "negative_weight": args.negative_weight, "C": args.c, "vocabulary": len(vectorizer.vocabulary_), "test_labels_used": False}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
