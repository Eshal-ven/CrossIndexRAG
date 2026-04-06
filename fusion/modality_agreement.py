"""
Modality agreement: hallucination control layer.

If the top-K results contain evidence from at least `require_modalities`
distinct modalities (text, image, table), we flag the response as "safe"
because multiple independent sources agree.
"""
from typing import List, Dict, Any


def check_agreement(results: List[Dict[str, Any]], require_modalities: int = 2) -> bool:
    """
    Returns True if `results` contains at least `require_modalities`
    distinct modality types.
    """
    modalities = {r.get("modality") for r in results if r.get("modality")}
    return len(modalities) >= require_modalities


def modality_breakdown(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Returns a count of results per modality — useful for logging and eval.
    e.g. {'text': 3, 'image': 1, 'table': 1}
    """
    breakdown: Dict[str, int] = {}
    for r in results:
        mod = r.get("modality", "unknown")
        breakdown[mod] = breakdown.get(mod, 0) + 1
    return breakdown