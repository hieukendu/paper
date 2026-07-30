from __future__ import annotations

from dataclasses import dataclass

from vipragsent.data.schema import PRAGMATIC_LABELS


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    label: str
    requires_finetuning: bool
    notes: str = ""


Q1_SYSTEMS = [
    SystemSpec("phobert_finetune", "PhoBERT fine-tune", True, "Default encoder fine-tune baseline."),
    SystemSpec("xlmr_large", "XLM-R-large", True, "Multilingual encoder fine-tune baseline."),
    SystemSpec("sailor_7b_sft", "Sailor-7B SFT", True, "QLoRA/SFT system; prediction import only here."),
    SystemSpec("vistral_7b_sft", "Vistral-7B SFT", True, "QLoRA/SFT system; prediction import only here."),
    SystemSpec("gpt41_mini_zero_shot", "GPT-4.1-mini zero-shot", False, "API or imported prediction baseline."),
    SystemSpec("gpt41_mini_8_shot", "GPT-4.1-mini 8-shot", False, "API or imported prediction baseline."),
    SystemSpec("vipragsent_full", "ViPragSent (ours)", True, "Full multitask model; prediction import only here."),
]

SMOKE_SYSTEM = SystemSpec(
    "heuristic_local",
    "Local heuristic smoke baseline",
    False,
    "No-train sanity baseline; not a paper system.",
)

SYSTEM_LABELS = {system.system_id: system.label for system in [*Q1_SYSTEMS, SMOKE_SYSTEM]}
SYSTEM_REQUIRES_FINETUNING = {
    system.system_id: system.requires_finetuning for system in [*Q1_SYSTEMS, SMOKE_SYSTEM]
}

ORDINARY_DATASETS = ["uit_vsfc", "uit_vsmec", "aivivn_2019"]
ORDINARY_SYSTEM_IDS = ["phobert_finetune", "xlmr_large", "vipragsent_full"]
# The adjudicated train split has 1,390 sarcasm-positive records.  A 2,048
# positive budget would require sampling with replacement, so it is excluded
# from the real-data protocol.
LOW_RESOURCE_BUDGETS = [64, 128, 256, 512, 1024]
LOW_RESOURCE_SYSTEM_IDS = ["phobert_finetune", "vipragsent_full"]
SEEDS = [20260520, 20260521, 20260522]

ABLATION_ROWS = [
    "full",
    "no_emotion_auxiliary",
    "no_ordinary_sentiment_auxiliary",
    "no_explanation_cot_auxiliary",
    "no_multitask",
    "no_task_uncertainty_weighting",
]

PRAGMATIC_POLARITY_CLASSES = [
    "positive",
    "negative",
    "neutral",
    "ironic-positive",
    "sarcastic-negative",
    "mocking",
]

METRIC_FIELDS = [*PRAGMATIC_LABELS, "macro_pragmatic_f1"]
