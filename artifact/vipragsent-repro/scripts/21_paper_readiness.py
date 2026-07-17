from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def has_model_archive_registry() -> bool:
    path = ROOT / "configs" / "artifact_registry.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        repositories = payload["external_model_archives"]["repositories"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    required = {"encoder_checkpoints", "sailor_7b_qlora", "vistral_7b_qlora"}
    return required.issubset(repositories) and all(
        isinstance(repositories[name], dict)
        and bool(repositories[name].get("repo_id"))
        and bool(repositories[name].get("url"))
        for name in required
    )


def main() -> int:
    checks = {
        "adjudicated_gold": {"complete": exists("data/processed/vipragsent_test.jsonl") and exists("data/manifest/gold_build_report.json")},
        "three_seed_main_encoders": {"complete": all(exists(f"results/predictions/main_pragmatic/{system}/{seed}.jsonl") for system in ("phobert_finetune", "vipragsent_full", "xlmr_large") for seed in (20260520, 20260521, 20260522))},
        "single_seed_7b_baselines": {
            "complete": all(exists(f"results/predictions/main_pragmatic/{system}/20260520.jsonl") for system in ("sailor_7b_sft", "vistral_7b_sft")),
            "note": "The repository protocol explicitly reports 7B baselines as single-seed, separately from the three-seed encoder protocol.",
        },
        "requested_three_seed_7b_extension": {
            "complete": all(exists(f"results/predictions/main_pragmatic/{system}/{seed}.jsonl") for system in ("sailor_7b_sft", "vistral_7b_sft") for seed in (20260520, 20260521, 20260522)),
            "note": "User-requested extension beyond the original single-seed 7B protocol.",
        },
        "iaa": {
            "complete": exists("results/annotation_agreement.json"),
            "note": "Agreement has been computed; report the observed values and adjudication protocol without overstating reliability.",
        },
        "paired_significance": {"complete": exists("results/significance.json")},
        "checkpoint_archive": {
            "complete": has_model_archive_registry(),
            "note": "Private Hugging Face archive URLs and reviewer access policy are recorded in configs/artifact_registry.json; Git intentionally excludes the weight files.",
        },
        "reproducibility_registry": {
            "complete": has_model_archive_registry(),
            "note": "The portable answer bundle copies the source-repository and model-archive registry without copying credentials.",
        },
        "private_research_terms": {"complete": exists("../../LICENSE"), "note": "Project materials are restricted to private non-commercial research; raw text redistribution is prohibited."},
        "visobert_public_release_permission": {"complete": False, "human_action": "Required only before a public raw-text dataset release; archive permission from the source rights holder."},
        "external_benchmark_provenance": {"complete": exists("configs/data_governance.yaml"), "note": "UIT datasets and AIVIVN are evaluation-only, not ViPragSent sources."},
        "rationale_faithfulness_audit": {"complete": True, "note": "Waived by the final protocol. Generated rationales must not be described as manually faithfulness-verified."},
        "paper_manuscript": {"complete": bool(list(ROOT.rglob("*.tex"))), "human_action": "Write the manuscript only from generated JSON/tables and cite no inherited main.pdf values."},
    }
    report = {"status": "complete" if all(row.get("complete") for row in checks.values()) else "action_required", "checks": checks}
    output = ROOT / "results/paper_readiness.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
