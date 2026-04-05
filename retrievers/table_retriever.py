from typing import List, Dict, Any

from embedders.table_embedder import TableEmbedder
from vector_db.table_index import TableIndex
from utils.logger import get_logger

logger = get_logger("table_retriever")


class TableRetriever:
    def __init__(self, index: TableIndex = None, embedder: TableEmbedder = None):
        self.index = index or TableIndex()
        self.embedder = embedder or TableEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.count() == 0:
            logger.warning("TableIndex is empty — returning no results.")
            return []
        # Use the SAME table embedder (MiniLM 384-dim) for queries too
        query_vec = self.embedder.embed(query)
        return self.index.query(query_vec, top_k=top_k)
