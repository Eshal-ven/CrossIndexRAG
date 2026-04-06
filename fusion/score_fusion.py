"""
Score fusion: combines embedding similarity score with cross-encoder score
using a weighted linear combination after min-max normalisation.
"""
from typing import List, Dict, Any

import numpy as np


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def fuse(candidates: List[Dict[str, Any]], alpha: float = 0.35) -> List[Dict[str, Any]]:
    """
    Fuse embedding score and cross-encoder score into a single `final_score`.

    Parameters
    ----------
    candidates : list of dicts, each must have 'score' and 'ce_score'
    alpha      : weight for embedding score (1-alpha goes to ce_score)

    Returns
    -------
    Same list, each item enriched with 'final_score', sorted descending.
    """
    if not candidates:
        return []

    embed_scores = np.array([c.get("score", 0.0)    for c in candidates], dtype=float)
    ce_scores    = np.array([c.get("ce_score", 0.0) for c in candidates], dtype=float)

    fused = alpha * _minmax_norm(embed_scores) + (1.0 - alpha) * _minmax_norm(ce_scores)

    for candidate, fs in zip(candidates, fused):
        candidate["final_score"] = float(fs)

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)