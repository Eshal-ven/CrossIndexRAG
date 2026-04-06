
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from configs.config import CROSS_ENCODER_MODEL
from utils.logger import get_logger

logger = get_logger("crossencoder_reranker")


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score each candidate with the cross-encoder and attach `ce_score`.
        Returns the list sorted by ce_score descending.

        Key used is `ce_score` — matches hybridRetrievalDemo.py exactly.
        """
        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["ce_score"] = float(score)

        return sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
