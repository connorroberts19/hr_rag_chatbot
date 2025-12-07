"""Configuration settings for the HR RAG Chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Embedding settings (local sentence-transformers)
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Retrieval settings
TOP_K_RESULTS = 5

# LLM settings (Ollama - local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
LLM_TEMPERATURE = 0.1
