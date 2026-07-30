from __future__ import annotations

from statistics import mean, pstdev


def aggregate_seed_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}
