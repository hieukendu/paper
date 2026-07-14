from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

VALID_STATUSES = {"pending", "mock", "complete", "partial", "blocked", "failed"}


def make_pending_result(experiment: str, *, source: str = "setup_only_dryrun") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment": experiment,
        "status": "pending",
        "metrics": {},
        "confidence_interval": None,
        "cost": None,
        "seed": None,
        "provenance": {
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "Pending scaffold only; not a real experimental result.",
        },
    }


def validate_result_schema(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "experiment", "status", "metrics", "provenance"):
        if key not in result:
            errors.append(f"missing {key}")
    if result.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}")
    if not isinstance(result.get("metrics", {}), dict):
        errors.append("metrics must be an object")
    if not isinstance(result.get("provenance", {}), dict):
        errors.append("provenance must be an object")
    return errors
