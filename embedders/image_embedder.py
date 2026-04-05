import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from configs.config import IMAGE_EMBED_MODEL
from utils.logger import get_logger

logger = get_logger("image_embedder")


class ImageEmbedder:
    def __init__(self, model_name: str = IMAGE_EMBED_MODEL):
        logger.info(f"Loading image embedder: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
        return vec / norm

    def embed(self, image_path: str) -> np.ndarray:
        """
        Embed a single image from disk.
        Returns shape (dim,).
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        vec = feats.cpu().numpy().reshape(-1)
        return self._normalize(vec)

    def embed_text_query(self, query: str) -> np.ndarray:
        """
        Embed a text query into CLIP's shared space so it can be compared
        against image embeddings at retrieval time.
        Returns shape (dim,).
        """
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        vec = feats.cpu().numpy().reshape(-1)
        return self._normalize(vec)