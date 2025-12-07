"""Document loader for markdown and text files."""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.config import RAW_DATA_DIR


def load_all_documents(directory: Optional[Path] = None) -> list[Document]:
    """Load all markdown and text documents from a directory."""
    if directory is None:
        directory = RAW_DATA_DIR

    documents = []

    # Load markdown files
    md_loader = DirectoryLoader(
        str(directory),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    try:
        md_docs = md_loader.load()
        if md_docs:
            print(f"Loaded {len(md_docs)} Markdown document(s)")
            documents.extend(md_docs)
    except Exception as e:
        print(f"Warning: Could not load Markdown files: {e}")

    # Load text files
    txt_loader = DirectoryLoader(
        str(directory),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    try:
        txt_docs = txt_loader.load()
        if txt_docs:
            print(f"Loaded {len(txt_docs)} Text document(s)")
            documents.extend(txt_docs)
    except Exception as e:
        print(f"Warning: Could not load Text files: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents
