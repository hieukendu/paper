from __future__ import annotations


def expected_calibration_error(confidences: list[float], correct: list[int], *, bins: int = 10) -> dict:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal lengths")
    if not confidences:
        return {"ece": 0.0, "bins": []}
    bucketed = []
    ece = 0.0
    n = len(confidences)
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        members = [
            (conf, ok)
            for conf, ok in zip(confidences, correct)
            if (low <= conf < high) or (idx == bins - 1 and conf == 1.0)
        ]
        if members:
            avg_conf = sum(conf for conf, _ in members) / len(members)
            acc = sum(ok for _, ok in members) / len(members)
            ece += (len(members) / n) * abs(acc - avg_conf)
        else:
            avg_conf = 0.0
            acc = 0.0
        bucketed.append({"low": low, "high": high, "n": len(members), "conf": avg_conf, "acc": acc})
    return {"ece": ece, "bins": bucketed}
