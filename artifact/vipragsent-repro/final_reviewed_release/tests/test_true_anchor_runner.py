import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_true_anchor_arbiter_cycle import load_spec


def test_runner_rejects_proxy_anchor_name(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"targets": {"irony": {"anchor_name": "incumbent"}}}))
    with pytest.raises(ValueError, match="sailor_7b_sft_qlora"):
        load_spec(path)


def test_runner_accepts_real_anchor_contract(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"targets": {"irony": {"anchor_name": "sailor_7b_sft_qlora"}, "idiom_figurative": {"anchor_name": "phobert_anchor+xlmr_large_anchor"}, "code_switching": {"anchor_name": "vistral_7b_sft_qlora"}}}))
    assert load_spec(path)["targets"]["code_switching"]["anchor_name"] == "vistral_7b_sft_qlora"
