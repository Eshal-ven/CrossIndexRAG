"""
Retrieval evaluation metrics.

All functions are stateless and operate on plain Python lists / numpy arrays
so they can be called from anywhere without importing the full pipeline.
"""
from typing import List

import numpy as np


# ── Precision & Recall ────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


# ── nDCG ─────────────────────────────────────────────────────────────────────

def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum((2 ** relevances - 1) / discounts))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.
    Binary relevance: 1 if the retrieved id is in relevant_ids, else 0.
    """
    relevances = np.array([1.0 if rid in relevant_ids else 0.0
                           for rid in retrieved_ids[:k]])
    ideal = np.ones(min(len(relevant_ids), k))   # best possible ordering
    idcg = _dcg_at_k(ideal, k)
    if idcg < 1e-9:
        return 0.0
    return _dcg_at_k(relevances, k) / idcg


# ── MRR ───────────────────────────────────────────────────────────────────────

def mean_reciprocal_rank(retrieved_ids_list: List[List[str]],
                         relevant_ids_list: List[List[str]]) -> float:
    """
    Mean Reciprocal Rank across multiple queries.

    Parameters
    ----------
    retrieved_ids_list : list of retrieved-id lists, one per query
    relevant_ids_list  : list of relevant-id sets,   one per query
    """
    rr_scores = []
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        rr = 0.0
        for rank, rid in enumerate(retrieved, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


# ── Modality coverage ─────────────────────────────────────────────────────────

def modality_coverage(results: List[dict]) -> float:
    """
    Fraction of the three modalities (text, image, table) represented
    in `results`. 1.0 means all three appear.
    """
    found = {r.get("modality") for r in results} & {"text", "image", "table"}
    return len(found) / 3.0