from typing import List, Dict, Any

from embedders.image_embedder import ImageEmbedder
from vector_db.image_index import ImageIndex
from utils.logger import get_logger

logger = get_logger("image_retriever")


class ImageRetriever:
    def __init__(self, index: ImageIndex = None, embedder: ImageEmbedder = None):
        self.index = index or ImageIndex()
        self.embedder = embedder or ImageEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the text query into CLIP's shared space, then searches
        the image index — enabling text-to-image retrieval.
        """
        if self.index.count() == 0:
            logger.warning("ImageIndex is empty — returning no results.")
            return []
        query_vec = self.embedder.embed_text_query(query)
        return self.index.query(query_vec, top_k=top_k)
