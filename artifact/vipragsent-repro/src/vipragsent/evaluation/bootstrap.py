from __future__ import annotations

import random
from statistics import mean
from typing import Callable


def bootstrap_ci(
    values: list,
    metric_fn: Callable[[list], float] = mean,
    *,
    resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 20260520,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(float(metric_fn(sample)))
    estimates.sort()
    alpha = (1 - ci) / 2
    lo = estimates[int(alpha * (len(estimates) - 1))]
    hi = estimates[int((1 - alpha) * (len(estimates) - 1))]
    return lo, hi
