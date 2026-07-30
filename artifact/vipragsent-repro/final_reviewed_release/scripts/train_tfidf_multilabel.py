from __future__ import annotations

"""Train a reproducible lexical multi-label classifier for ViPragSent.

Character features are deliberately included because code-switching, emoji,
creative spellings and Vietnamese teencode carry useful surface signals that
subword encoders may smooth away.  Training consumes only ``--train``;
threshold fitting is a separate development-only operation.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl


def load(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def matrix(word, char, texts: list[str]):
    return hstack([word.transform(texts), char.transform(texts)], format="csr")


def fit(args: argparse.Namespace) -> None:
    rows = load(args.train)
    texts = [row["text"] for row in rows]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.995, sublinear_tf=True, strip_accents=None, max_features=args.max_word_features)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, sublinear_tf=True, max_features=args.max_char_features)
    features = hstack([word.fit_transform(texts), char.fit_transform(texts)], format="csr")
    classifiers = {}
    for label in PRAGMATIC_LABELS:
        y = [int(row["labels"][label]) for row in rows]
        if args.classifier == "logreg":
            classifier = LogisticRegression(C=args.c, class_weight="balanced", max_iter=args.max_iter, solver="liblinear", random_state=args.seed)
        else:
            classifier = LinearSVC(C=args.c, class_weight="balanced", max_iter=args.max_iter, random_state=args.seed)
        classifiers[label] = classifier.fit(features, y)
    payload = {"word": word, "char": char, "classifiers": classifiers, "labels": list(PRAGMATIC_LABELS), "args": vars(args)}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model)
    print(json.dumps({"status": "ok", "train_records": len(rows), "features": int(features.shape[1]), "model": str(args.model)}, ensure_ascii=False, indent=2, default=str))


def predict(args: argparse.Namespace) -> None:
    payload = joblib.load(args.model)
    rows = load(args.data)
    features = matrix(payload["word"], payload["char"], [row["text"] for row in rows])
    probabilities = {}
    for label in PRAGMATIC_LABELS:
        classifier = payload["classifiers"][label]
        if hasattr(classifier, "predict_proba"):
            probabilities[label] = classifier.predict_proba(features)[:, 1]
        else:
            # A monotonic sigmoid preserves every possible thresholded SVM
            # decision while retaining the common prediction-file schema.
            values = classifier.decision_function(features)
            probabilities[label] = 1.0 / (1.0 + __import__("numpy").exp(-values))
    output = []
    for index, row in enumerate(rows):
        values = {label: float(probabilities[label][index]) for label in PRAGMATIC_LABELS}
        output.append({"id": str(row["id"]), "system": args.system, "seed": args.seed, "predictions": {label: int(values[label] >= 0.5) for label in PRAGMATIC_LABELS}, "probabilities": values})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    print(json.dumps({"status": "ok", "records": len(output), "output": str(args.output)}, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    fit_parser.add_argument("--model", type=Path, required=True)
    fit_parser.add_argument("--c", type=float, default=1.0)
    fit_parser.add_argument("--classifier", choices=("logreg", "linear_svc"), default="logreg")
    fit_parser.add_argument("--seed", type=int, default=20260724)
    fit_parser.add_argument("--max-iter", type=int, default=400)
    fit_parser.add_argument("--max-word-features", type=int, default=120000)
    fit_parser.add_argument("--max-char-features", type=int, default=180000)
    fit_parser.set_defaults(func=fit)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--data", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--system", required=True)
    predict_parser.add_argument("--seed", type=int, default=20260724)
    predict_parser.set_defaults(func=predict)
    args = parser.parse_args(); args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
