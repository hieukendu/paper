"""Validated, label-free probability sources used by fair-framework cycles."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbabilitySource:
    source_name: str
    architecture: str
    checkpoint_path: str
    checkpoint_sha256: str
    config_path: str
    tokenizer_path: str
    environment_path: str
    dev_probability_path: str
    test_probability_path: str
    predict_command: str
    status: str

    @classmethod
    def load(cls, path: Path) -> "ProbabilitySource":
        payload = json.loads(path.read_text())
        required = set(cls.__dataclass_fields__)
        missing = required - set(payload)
        if missing:
            raise ValueError(f"source manifest missing {sorted(missing)}")
        return cls(**{key: payload[key] for key in required})

    def verify_checkpoint(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.checkpoint_sha256:
            raise ValueError(f"checkpoint SHA mismatch for {self.source_name}")

    def verify_environment(self) -> None:
        if not Path(self.config_path).is_file() or not Path(self.environment_path).is_file():
            raise FileNotFoundError("missing source config or environment metadata")

    def probability_path(self, split: str) -> Path:
        if split == "dev": return Path(self.dev_probability_path)
        if split == "test": return Path(self.test_probability_path)
        raise ValueError(f"unknown split {split}")

    def validate_probabilities(self, split: str, ids: list[str], label: str) -> None:
        path = self.probability_path(split)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        found = {str(row["id"]): row for row in rows}
        if len(found) != len(rows) or set(found) != set(ids):
            raise ValueError(f"ID alignment failed for {self.source_name}/{split}")
        if any(row.get("label") != label or not 0 <= float(row.get("probability", -1)) <= 1 for row in rows):
            raise ValueError(f"invalid probabilities for {self.source_name}/{split}")

    def load_probabilities(self, split: str, ids: list[str], label: str) -> list[float]:
        self.validate_probabilities(split, ids, label)
        rows = {str(json.loads(line)["id"]): json.loads(line) for line in self.probability_path(split).read_text().splitlines() if line}
        return [float(rows[item]["probability"]) for item in ids]

    def predict(self, split: str, *, load_labels: bool = False) -> None:
        if load_labels:
            raise ValueError("ProbabilitySource inference is label-free by design")
        command = self.predict_command.format(split=split, output=self.probability_path(split))
        subprocess.run(command, shell=True, check=True)
