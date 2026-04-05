from typing import List, Dict, Any

from embedders.text_embedder import TextEmbedder
from vector_db.text_index import TextIndex
from utils.logger import get_logger

logger = get_logger("text_retriever")


class TextRetriever:
    def __init__(self, index: TextIndex = None, embedder: TextEmbedder = None):
        self.index = index or TextIndex()
        self.embedder = embedder or TextEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.count() == 0:
            logger.warning("TextIndex is empty — returning no results.")
            return []
        query_vec = self.embedder.embed(query)
        return self.index.query(query_vec, top_k=top_k)
