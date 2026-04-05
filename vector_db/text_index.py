from typing import Dict, List, Any
import numpy as np

from vector_db.chroma_client import get_client
from configs.config import TEXT_COLLECTION
from utils.logger import get_logger

logger = get_logger("text_index")


class TextIndex:
    def __init__(self, collection_name: str = TEXT_COLLECTION):
        self.collection = get_client().get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"TextIndex ready — collection: '{collection_name}', "
                    f"docs: {self.collection.count()}")

    def add(self, source_id: str, embedding: np.ndarray,
            content: str, metadata: Dict[str, Any] = None) -> None:
        self.collection.upsert(
            ids=[source_id],
            embeddings=[embedding.tolist()],
            documents=[content],
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
                "text": doc,
                "score": 1.0 - dist,   # cosine distance → similarity
                "metadata": meta,
                "modality": "text",
            })
        return items

    def count(self) -> int:
        return self.collection.count()
