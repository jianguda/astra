"""Aggregation helpers of the research-question runners."""
from __future__ import annotations


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def average(items: list):
    """Mean of numeric leaves over a list of equally shaped nested dictionaries."""
    if not items:
        return {}
    if isinstance(items[0], dict):
        return {key: average([item[key] for item in items if key in item]) for key in items[0]}
    return round(mean([float(item) for item in items]), 4)
