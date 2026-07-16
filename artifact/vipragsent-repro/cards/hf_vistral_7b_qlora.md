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

Exact run metadata is recorded in `run_manifest.json` and `history.json`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Viet-Mistral/Vistral-7B-Chat"
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, "Thundergod2007/vipragsent-vistral-7b-qlora", subfolder="adapter")
tokenizer = AutoTokenizer.from_pretrained(base_id)
```

Access to the base model may be gated. Use the exact prompt/parser in `scripts/train_qlora_sft.py` from the [ViPragSent repository](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro).

## Evaluation and provenance

Generated results are published in the repository's [`answer/`](https://github.com/hieukendu/paper/tree/main/artifact/vipragsent-repro/answer) evidence bundle. That bundle, rather than this card, is authoritative for metrics and confidence intervals while multi-seed runs are being completed.

## Limitations and responsible use

The model may fail to emit valid JSON and may reproduce social-media or base-model biases. Pragmatic labels are subjective and rare classes remain difficult. Do not use this artifact for surveillance, decisions about individuals, or high-impact automation without human review.

This adapter is a research artifact. The Vistral base-model license/access terms continue to apply. ViPragSent raw-text redistribution remains restricted pending confirmation of the source ViSoBERT export terms.
