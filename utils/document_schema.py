from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """Unified schema for all modalities stored and retrieved in CrossIndexRAG."""

    source_id: str
    modality: str                          # "text" | "image" | "table"
    content: str                           # text body, image path, or serialised table row
    embedding: Optional[Any] = None        # numpy array, set after embedding
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "modality": self.modality,
            "content": self.content,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Document":
        return Document(
            source_id=d["source_id"],
            modality=d["modality"],
            content=d["content"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class RetrievedResult:
    """Returned by every retriever."""

    source_id: str
    modality: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.source_id,
            "modality": self.modality,
            "text": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }