"""LLM configuration using Ollama (local)."""

from langchain_ollama import OllamaLLM

from src.config import LLM_TEMPERATURE, OLLAMA_MODEL


def get_llm():
    """Get the Ollama LLM for local inference."""
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
    )
