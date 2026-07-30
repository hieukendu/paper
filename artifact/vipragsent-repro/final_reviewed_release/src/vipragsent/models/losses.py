from __future__ import annotations

import math


def uncertainty_weighted_loss(losses: dict[str, float], log_variances: dict[str, float]) -> float:
    total = 0.0
    for name, loss in losses.items():
        log_var = log_variances.get(name, 0.0)
        sigma_sq = math.exp(log_var)
        total += (loss / (2 * sigma_sq)) + 0.5 * log_var
    return total
