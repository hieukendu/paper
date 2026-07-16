from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def main() -> int:
    checks = {
        "adjudicated_gold": {"complete": exists("data/processed/vipragsent_test.jsonl") and exists("data/manifest/gold_build_report.json")},
        "three_seed_main_encoders": {"complete": all(exists(f"results/predictions/main_pragmatic/{system}/{seed}.jsonl") for system in ("phobert_finetune", "vipragsent_full", "xlmr_large") for seed in (20260520, 20260521, 20260522))},
        "single_seed_7b_baselines": {
            "complete": all(exists(f"results/predictions/main_pragmatic/{system}/20260520.jsonl") for system in ("sailor_7b_sft", "vistral_7b_sft")),
            "note": "The repository protocol explicitly reports 7B baselines as single-seed, separately from the three-seed encoder protocol.",
        },
        "iaa": {"complete": exists("results/annotation_agreement.json")},
        "paired_significance": {"complete": exists("results/significance.json")},
        "checkpoint_archive": {"complete": len(list((ROOT / "outputs").rglob("best.pt"))) == 23 and len(list((ROOT / "outputs").rglob("adapter_model.safetensors"))) == 2},
        "visobert_release_license": {"complete": False, "human_action": "Confirm the ViSoBERT export's license/redistribution terms."},
        "external_benchmark_provenance": {"complete": exists("configs/data_governance.yaml"), "note": "UIT datasets and AIVIVN are evaluation-only, not ViPragSent sources."},
        "rationale_faithfulness_audit": {"complete": False, "human_action": "The current waiver is not a substitute for the planned >=5% manual audit."},
        "paper_manuscript": {"complete": bool(list(ROOT.rglob("*.tex"))), "human_action": "Write the manuscript only from generated JSON/tables and cite no inherited main.pdf values."},
    }
    report = {"status": "complete" if all(row.get("complete") for row in checks.values()) else "action_required", "checks": checks}
    output = ROOT / "results/paper_readiness.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
