import chromadb
from chromadb.config import Settings

from configs.config import CHROMA_PERSIST_DIR
from utils.logger import get_logger

logger = get_logger("chroma_client")

_client = None


def get_client() -> chromadb.ClientAPI:
    """Return a singleton persistent ChromaDB client."""
    global _client
    if _client is None:
        logger.info(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client