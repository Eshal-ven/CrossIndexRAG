import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

from configs.config import TEXT_EMBED_MODEL
from utils.logger import get_logger

logger = get_logger("text_embedder")


class TextEmbedder:
    def __init__(self, model_name: str = TEXT_EMBED_MODEL):
        logger.info(f"Loading text embedder: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
        return vec / norm

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one string or a list of strings.
        Returns shape (dim,) for a single string, (N, dim) for a list.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = self._normalize(vecs)
        return vecs[0] if single else vecs