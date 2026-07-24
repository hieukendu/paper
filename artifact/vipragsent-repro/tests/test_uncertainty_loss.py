import importlib.util
from pathlib import Path

import torch
from torch import nn


TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "train_multitask_encoder.py"
SPEC = importlib.util.spec_from_file_location("train_multitask_encoder", TRAINER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))
        self.logits = nn.Parameter(torch.zeros(2, 6))
        self.polarity_logits = nn.Parameter(torch.zeros(2, 3))
        self.emotion_logits = nn.Parameter(torch.zeros(2, 7))

    def forward(self, input_ids, attention_mask):
        pooled = torch.zeros(2, 4)
        return self.logits, self.polarity_logits, self.emotion_logits, pooled


def test_disabled_uncertainty_tasks_receive_no_log_var_gradient():
    model = TinyModel()
    batch = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "pragmatic": torch.tensor([[1.0] * 6, [0.0] * 6]),
        "polarity": torch.tensor([0, 1]),
        "emotion": torch.tensor([0, 1]),
    }
    loss, _, _ = MODULE.task_loss(
        model, batch, torch.device("cpu"), beta=0.3, auxiliary_weight=1.0,
        focal_gamma=0.0, pragmatic_label_weights=None, pragmatic_pos_weight=None,
        use_emotion=False, use_polarity=False, use_rationale=False,
    )
    loss.backward()
    assert model.log_vars.grad[0].abs().item() > 0
    assert torch.equal(model.log_vars.grad[1:], torch.zeros(3))
