"""Embedding model configuration using local sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import LOCAL_EMBEDDING_MODEL


def get_embedding_model():
    """Get the local embedding model (sentence-transformers)."""
    return HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
