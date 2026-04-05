from typing import Dict, List, Any
import numpy as np

from vector_db.chroma_client import get_client
from configs.config import IMAGE_COLLECTION
from utils.logger import get_logger

logger = get_logger("image_index")


class ImageIndex:
    def __init__(self, collection_name: str = IMAGE_COLLECTION):
        self.collection = get_client().get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ImageIndex ready — collection: '{collection_name}', "
                    f"docs: {self.collection.count()}")

    def add(self, source_id: str, embedding: np.ndarray,
            image_path: str, metadata: Dict[str, Any] = None) -> None:
        self.collection.upsert(
            ids=[source_id],
            embeddings=[embedding.tolist()],
            documents=[image_path],          # store path as the "document"
            metadatas=[metadata or {}],
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for sid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({
                "id": sid,
                "text": doc,             # image path — used as content downstream
                "score": 1.0 - dist,
                "metadata": meta,
                "modality": "image",
            })
        return items

    def count(self) -> int:
        return self.collection.count()