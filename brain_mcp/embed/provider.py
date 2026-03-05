"""
brain-mcp — Embedding provider abstraction.

Provides an ABC for embedding providers and a FastEmbed implementation
using BAAI/bge-small-en-v1.5 (33M params, 384d, ~107MB).

This replaces the old sentence-transformers dependency with the much
lighter fastembed library.
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Returns embedding vector."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class FastEmbedProvider(EmbeddingProvider):
    """Embedding provider using fastembed (ONNX-based, lightweight).

    Uses BAAI/bge-small-en-v1.5 by default:
    - 33M parameters
    - 384 dimensions
    - ~107MB download
    - No PyTorch dependency
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        # Determine dimension from a test embed
        test = list(self._model.embed(["test"]))
        self._dim = len(test[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using fastembed."""
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using fastembed's query_embed (optimized for queries)."""
        results = list(self._model.query_embed(text))
        emb = results[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

    @property
    def dimension(self) -> int:
        return self._dim


# Module-level singleton
_provider: Optional[EmbeddingProvider] = None


def get_provider(model_name: Optional[str] = None) -> EmbeddingProvider:
    """Get the cached embedding provider singleton.

    Args:
        model_name: Model name override. If None, uses config default.

    Returns:
        FastEmbedProvider instance.
    """
    global _provider
    if _provider is None:
        if model_name is None:
            from brain_mcp.config import get_config
            model_name = get_config().embedding.model
        _provider = FastEmbedProvider(model_name=model_name)
    return _provider


def reset_provider():
    """Reset the cached provider (useful for testing)."""
    global _provider
    _provider = None
