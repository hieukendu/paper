---
library_name: pytorch
license: other
language:
- vi
pipeline_tag: text-classification
tags:
- phobert
- xlm-roberta
- multi-task-learning
- sentiment-analysis
- sarcasm-detection
- vietnamese
---

# ViPragSent Experiment Checkpoints

Archive: [Thundergod2007/vipragsent-experiment-checkpoints](https://huggingface.co/Thundergod2007/vipragsent-experiment-checkpoints) (private before the associated submission decision).

This private archival repository contains completed encoder checkpoints for the ViPragSent experiments. It mirrors the `outputs/` hierarchy: each run directory contains `best.pt`, `history.json`, and `run_manifest.json`.

## Included systems

- PhoBERT pragmatic single-task/fine-tuning baseline.
- ViPragSent full multi-task PhoBERT model.
- XLM-R-large multi-task model.
- Registered PhoBERT ablations: no rationale auxiliary, no emotion auxiliary, no polarity auxiliary, and no task-uncertainty weighting.
- Low-resource sarcasm runs at the feasible registered budgets.

The repository does not include base-model weights, raw or processed dataset text, API credentials, predictions, or caches.

## Training protocol

Main encoder experiments use seeds `20260520`, `20260521`, and `20260522`, maximum sequence length 128, BF16 on an H100 20 GB MIG allocation, early stopping, and checkpoint selection by development macro pragmatic F1. Run manifests are authoritative for the actual epochs, flags, elapsed time, base-model path, and output provenance of each run.

## Loading

These are project-native PyTorch checkpoints rather than standalone Transformers `save_pretrained` exports. Reconstruct `MultiTaskModel` using the matching base model and flags, then load the checkpoint:

```python
import torch
from scripts.train_multitask_encoder import MultiTaskModel

model = MultiTaskModel(BASE_MODEL_ID, uncertainty=True, rationale_aux=True)
payload = torch.load("vipragsent_full/20260520/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"])
model.eval()
```

Use the corresponding `run_manifest.json` before choosing `uncertainty` or `rationale_aux`; ablation checkpoints require different constructor flags. Full prediction commands are implemented in `scripts/train_multitask_encoder.py` in the [ViPragSent repository](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro).

## Results and reproducibility

Predictions, seed aggregates, bootstrap confidence intervals, paired significance tests, calibration, low-resource results, IAA, and provenance are versioned in the repository's [`answer/`](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro/answer) bundle. Values in that bundle are generated from prediction JSONL and manifests; earlier manuscript values are not treated as evidence.

## Limitations and licenses

The checkpoints inherit limitations and applicable terms from their base models (`vinai/phobert-base` and `FacebookAI/xlm-roberta-large`). They may encode social-media bias, label subjectivity, and class imbalance. They are not suitable for surveillance or high-impact decisions about individuals.

Raw ViPragSent text is not distributed here. Its redistribution remains restricted pending confirmation of the source ViSoBERT export terms.
