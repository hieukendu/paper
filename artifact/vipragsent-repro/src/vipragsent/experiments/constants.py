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
    SystemSpec("phobert_single_task", "PhoBERT (single-task)", True, "One head per pragmatic label."),
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
# The adjudicated train split has 1,390 sarcasm-positive records.  A 2,048
# positive budget would require sampling with replacement, so it is excluded
# from the real-data protocol.
LOW_RESOURCE_BUDGETS = [64, 128, 256, 512, 1024]
SEEDS = [20260520, 20260521, 20260522, 20260523, 20260524]

ABLATION_ROWS = [
    "full",
    "no_emotion_auxiliary",
    "no_ordinary_sentiment_auxiliary",
    "no_explanation_cot_auxiliary",
    "no_multitask",
    "no_task_uncertainty_weighting",
    "explanation_augmented_inference",
    "hard_label_distillation",
]

PRAGMATIC_POLARITY_CLASSES = [
    "positive",
    "negative",
    "neutral",
    "ironic-positive",
    "sarcastic-negative",
    "mocking",
]

COST_BREAKDOWN = {
    "assumptions": {
        "source": "Paper protocol estimates; not measured by this no-fine-tune runner.",
        "annotation_cost_usd": 1860.0,
        "gpu": "A100",
        "compute_price_usd_per_gpu_hour": 0.80,
        "api_model": "gpt-4.1-mini",
    },
    "systems": {
        "phobert_finetune": {"gpu_hours": 8.4, "compute_usd": 6.7, "api_usd": 0.0, "total_usd": 1866.7},
        "xlmr_large": {"gpu_hours": 17.1, "compute_usd": 13.7, "api_usd": 0.0, "total_usd": 1873.7},
        "sailor_7b_sft": {"gpu_hours": 41.2, "compute_usd": 33.0, "api_usd": 0.0, "total_usd": 1893.0},
        "vistral_7b_sft": {"gpu_hours": 38.6, "compute_usd": 30.9, "api_usd": 0.0, "total_usd": 1890.9},
        "gpt41_mini_zero_shot": {
            "gpu_hours": 0.0,
            "compute_usd": 0.0,
            "api_usd": None,
            "total_usd": None,
            "note": "Measure from Azure usage logs; pricing is not hard-coded.",
        },
        "gpt41_mini_8_shot": {
            "gpu_hours": 0.0,
            "compute_usd": 0.0,
            "api_usd": None,
            "total_usd": None,
            "note": "Measure from Azure usage logs; pricing is not hard-coded.",
        },
        "vipragsent_phobert": {"gpu_hours": 12.6, "compute_usd": 10.1, "api_usd": 0.0, "total_usd": 1870.1},
        "vipragsent_full": {"gpu_hours": 46.8, "compute_usd": 37.4, "api_usd": 0.0, "total_usd": 1897.4},
    },
}

METRIC_FIELDS = [*PRAGMATIC_LABELS, "macro_pragmatic_f1"]
