import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Union

from configs.config import TABLE_EMBED_MODEL
from utils.logger import get_logger

logger = get_logger("table_embedder")


class TableEmbedder:
    def __init__(self, model_name: str = TABLE_EMBED_MODEL):
        logger.info(f"Loading table embedder: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
        return vec / norm

    def _row_to_text(self, row: Union[Dict, str]) -> str:
        """Convert a dict row or plain string into a flat text representation."""
        if isinstance(row, dict):
            return " | ".join(f"{k}: {v}" for k, v in row.items())
        return str(row)

    def embed(self, row: Union[Dict, str]) -> np.ndarray:
        """
        Embed a single table row (dict or pre-serialised string).
        Returns shape (dim,).
        """
        text = self._row_to_text(row)
        vec = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return self._normalize(vec)