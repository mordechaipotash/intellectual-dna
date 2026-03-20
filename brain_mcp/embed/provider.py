"""
brain-mcp — Embedding provider abstraction.

Provides an ABC for embedding providers with two implementations:

1. FastEmbedProvider (default) — BAAI/bge-small-en-v1.5
   33M params, 384d, ~107MB. No PyTorch dependency.

2. SentenceTransformerProvider (legacy fallback)
   For users with existing 768d vectors who haven't rebuilt yet.
   Requires sentence-transformers + PyTorch (~1.3GB).

Auto-detection order:
  1. fastembed installed → FastEmbedProvider
  2. sentence-transformers installed → SentenceTransformerProvider
  3. Neither → ImportError with helpful message
"""

from abc import ABC, abstractmethod
from typing import Optional
import sys


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

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for CLI output."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string."""
        ...


class FastEmbedProvider(EmbeddingProvider):
    """Embedding provider using fastembed (ONNX-based, lightweight).

    Uses BAAI/bge-small-en-v1.5 by default:
    - 33M parameters
    - 384 dimensions
    - ~107MB download
    - No PyTorch dependency
    """

    # Default model for fastembed — lightweight and fast
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        from fastembed import TextEmbedding

        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = TextEmbedding(model_name=self._model_name)
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

    @property
    def provider_name(self) -> str:
        return f"fastembed ({self._model_name}, {self._dim} dims)"

    @property
    def model_name(self) -> str:
        return self._model_name


class SentenceTransformerProvider(EmbeddingProvider):
    """Legacy embedding provider using sentence-transformers.

    Requires PyTorch (~1.3GB). Use only if you have existing 768d vectors
    and haven't rebuilt yet, or need a specific sentence-transformers model.

    Install: pip install 'brain-mcp[embed-torch]'
    """

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = SentenceTransformer(
            self._model_name, trust_remote_code=True
        )
        # Determine dimension from a test embed
        test = self._model.encode(["test"])
        self._dim = test.shape[1]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using sentence-transformers."""
        if not texts:
            return []
        # Nomic models expect a prefix for document embedding
        if "nomic" in self._model_name.lower():
            texts = [f"search_document: {t}" for t in texts]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using sentence-transformers."""
        if "nomic" in self._model_name.lower():
            text = f"search_query: {text}"
        emb = self._model.encode([text], show_progress_bar=False)[0]
        return emb.tolist()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"sentence-transformers ({self._model_name}, {self._dim} dims)"

    @property
    def model_name(self) -> str:
        return self._model_name


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER AUTO-DETECTION & SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_provider: Optional[EmbeddingProvider] = None


def _detect_available_provider() -> str:
    """Detect which embedding backend is available.

    Returns 'fastembed', 'sentence-transformers', or raises ImportError.
    """
    # Try fastembed first (preferred — lightweight)
    try:
        import fastembed  # noqa: F401
        return "fastembed"
    except ImportError:
        pass

    # Fall back to sentence-transformers (legacy — heavy)
    try:
        import sentence_transformers  # noqa: F401
        return "sentence-transformers"
    except ImportError:
        pass

    # Detect if running in a pipx environment
    import os
    in_pipx = "pipx" in (os.environ.get("VIRTUAL_ENV", "") + sys.prefix)
    if in_pipx:
        raise ImportError(
            "No embedding backend found. Install one of:\n"
            "  pipx inject brain-mcp fastembed              # recommended, ~107MB\n"
            "  pipx inject brain-mcp sentence-transformers  # legacy, ~1.3GB\n"
        )
    else:
        raise ImportError(
            "No embedding backend found. Install one of:\n"
            "  pip install 'brain-mcp[embed]'           # fastembed (recommended, ~107MB)\n"
            "  pip install 'brain-mcp[embed-torch]'     # sentence-transformers (legacy, ~1.3GB)\n"
        )


def get_provider(
    model_name: Optional[str] = None,
    force_provider: Optional[str] = None,
) -> EmbeddingProvider:
    """Get the cached embedding provider singleton.

    Args:
        model_name: Model name override. If None, uses provider default.
        force_provider: Force a specific provider ('fastembed' or 'sentence-transformers').
                       If None, auto-detects based on what's installed.

    Returns:
        EmbeddingProvider instance (cached as singleton).
    """
    global _provider
    if _provider is not None:
        return _provider

    if force_provider:
        backend = force_provider
    else:
        backend = _detect_available_provider()

    if backend == "fastembed":
        # For fastembed, use its default model unless explicitly overridden
        # Config might have the old nomic model name — ignore it for fastembed
        fastembed_model = model_name if model_name and "bge" in model_name.lower() else None
        _provider = FastEmbedProvider(model_name=fastembed_model)
    elif backend == "sentence-transformers":
        _provider = SentenceTransformerProvider(model_name=model_name)
    else:
        raise ValueError(f"Unknown provider: {backend}")

    print(f"Using {_provider.provider_name}", file=sys.stderr, flush=True)
    return _provider


def reset_provider():
    """Reset the cached provider (useful for testing)."""
    global _provider
    _provider = None
