"""Ranking metrics shared by the eval scripts."""

from __future__ import annotations

import math


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ranking_metrics(ranked: list[str], relevant: set[str], gain: dict[str, float], ns: tuple[int, ...]) -> dict:
    out = {}
    for n in ns:
        hits = ranked[:n]
        inter = [i for i in hits if i in relevant]
        out[f"recall@{n}"] = len(inter) / len(relevant) if relevant else 0.0
        out[f"precision@{n}"] = len(inter) / n
        out[f"hit@{n}"] = 1.0 if inter else 0.0
        ideal = sorted(gain.values(), reverse=True)[:n]
        denom = _dcg(ideal) if ideal else 0.0
        out[f"ndcg@{n}"] = (_dcg([gain.get(i, 0.0) for i in hits]) / denom) if denom else 0.0
    first = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
    out["mrr"] = (1.0 / first) if first else 0.0
    return out


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0
