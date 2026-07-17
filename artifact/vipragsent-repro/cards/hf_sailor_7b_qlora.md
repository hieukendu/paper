---
library_name: peft
license: other
base_model: sail/Sailor-7B
language:
- vi
pipeline_tag: text-classification
tags:
- qlora
- peft
- sentiment-analysis
- sarcasm-detection
- vietnamese
---

# ViPragSent Sailor-7B QLoRA Adapter

This private research repository contains QLoRA adapters for classifying Vietnamese social-media text under the ViPragSent schema. It does not contain the Sailor-7B base weights, dataset text, credentials, or Hugging Face caches.

## Tasks

The model emits one JSON object containing six binary pragmatic labels (`implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`, `code_switching`, and `mocking`), intended polarity, and emotion.

## Training protocol

- Base model: `sail/Sailor-7B`
- Method: NF4 4-bit QLoRA SFT
- LoRA rank/alpha/dropout: 16/32/0.05
- Epochs: 3
- Learning rate: 1e-4
- Maximum sequence length: 256 in the executable trainer
- Training split: 8,000 adjudicated ViPragSent records
- Evaluation split: 2,000 adjudicated ViPragSent records
- Hardware: NVIDIA H100 20 GB MIG allocation

Exact run-specific settings and elapsed time are recorded in `run_manifest.json` and `history.json`. Completed runs are stored under `seeds/20260520`, `seeds/20260521`, and `seeds/20260522`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "sail/Sailor-7B"
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, "Thundergod2007/vipragsent-sailor-7b-qlora", subfolder="seeds/20260520/adapter")
tokenizer = AutoTokenizer.from_pretrained(base_id)
```

Use the prompt and deterministic JSON parser from `scripts/train_qlora_sft.py` in the [ViPragSent repository](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro). Loading the adapter alone without the registered prompt/parser does not reproduce reported predictions.

## Evaluation

On the fixed 2,000-record adjudicated ViPragSent test split, macro pragmatic F1 across three seeds is **82.1918**, with a seed-level 95% CI of **[81.9447, 82.4390]**. Seed values are 82.3448, 81.9417, and 82.2890. These are percentage-scale F1 values generated from the prediction JSONL files.

## Provenance

Machine-readable metrics, predictions, bootstrap intervals, significance results, and data provenance are maintained in the repository's [`answer/`](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro/answer) bundle. That generated bundle is authoritative if this summary ever diverges.

## Limitations and responsible use

Outputs may reflect annotation subjectivity, class imbalance, social-media bias, and generation/JSON parsing failures. Do not use the model for decisions about individuals, surveillance, or fully automated high-impact moderation. Generated rationales used elsewhere in the project were not manually faithfulness-audited.

The adapter is a research artifact. Use of the base model remains governed by Sailor-7B's license and access terms. ViPragSent raw-text redistribution remains restricted pending confirmation of the source ViSoBERT export terms.
