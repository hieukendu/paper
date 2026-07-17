---
library_name: peft
license: other
base_model: Viet-Mistral/Vistral-7B-Chat
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

# ViPragSent Vistral-7B QLoRA Adapter

Archive: [Thundergod2007/vipragsent-vistral-7b-qlora](https://huggingface.co/Thundergod2007/vipragsent-vistral-7b-qlora) (private before the associated submission decision).

This private research repository contains QLoRA adapters for classifying Vietnamese social-media text under the ViPragSent schema. It excludes Vistral base weights, dataset text, credentials, and caches.

## Tasks

The registered prompt requests a JSON object with six binary pragmatic labels, intended polarity, and emotion. The executable prediction pipeline uses deterministic generation and records parse failures rather than silently treating malformed generations as valid evidence.

## Training protocol

- Base model: `Viet-Mistral/Vistral-7B-Chat`
- Method: NF4 4-bit QLoRA SFT
- Attention implementation: SDPA
- LoRA rank/alpha/dropout: 16/32/0.05
- Epochs: 3
- Learning rate: 1e-4
- Maximum sequence length: 256 in the executable trainer
- Training split: 8,000 adjudicated ViPragSent records
- Evaluation split: 2,000 adjudicated ViPragSent records
- Hardware: NVIDIA H100 20 GB MIG allocation

Exact run metadata is recorded in `run_manifest.json` and `history.json`. Completed runs are stored under `seeds/20260520`, `seeds/20260521`, and `seeds/20260522`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Viet-Mistral/Vistral-7B-Chat"
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, "Thundergod2007/vipragsent-vistral-7b-qlora", subfolder="seeds/20260520/adapter")
tokenizer = AutoTokenizer.from_pretrained(base_id)
```

Access to the base model may be gated. Use the exact prompt/parser in `scripts/train_qlora_sft.py` from the [ViPragSent repository](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro).

## Evaluation

On the fixed 2,000-record adjudicated ViPragSent test split, macro pragmatic F1 across three seeds is **82.8250**, with a seed-level 95% CI of **[82.6486, 83.0014]**. Seed values are 82.9462, 82.8796, and 82.6492. These are percentage-scale F1 values generated from the prediction JSONL files.

## Provenance

Generated results are published in the repository's [`answer/`](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro/answer) evidence bundle. That generated bundle is authoritative if this summary ever diverges.

## Limitations and responsible use

The model may fail to emit valid JSON and may reproduce social-media or base-model biases. Pragmatic labels are subjective and rare classes remain difficult. Do not use this artifact for surveillance, decisions about individuals, or high-impact automation without human review.

This adapter is a research artifact. The Vistral base-model license/access terms continue to apply. ViPragSent raw-text redistribution remains restricted pending confirmation of the source ViSoBERT export terms.
