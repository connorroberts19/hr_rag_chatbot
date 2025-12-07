"""Text processing and chunking utilities."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^\s*[-•]\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def chunk_markdown_by_headers(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split Markdown documents by headers first, then by size.

    This preserves semantic structure better for HR documents
    that are organized by sections.
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []

    for doc in documents:
        # First split by markdown headers
        md_chunks = markdown_splitter.split_text(doc.page_content)

        # Then split large sections by size
        for md_chunk in md_chunks:
            # Preserve original metadata and add header info
            metadata = doc.metadata.copy()
            metadata.update(md_chunk.metadata)

            if len(md_chunk.page_content) > chunk_size:
                # Further split if too large
                sub_chunks = text_splitter.split_text(md_chunk.page_content)
                for sub_chunk in sub_chunks:
                    all_chunks.append(Document(
                        page_content=clean_text(sub_chunk),
                        metadata=metadata,
                    ))
            else:
                all_chunks.append(Document(
                    page_content=clean_text(md_chunk.page_content),
                    metadata=metadata,
                ))

    # Filter empty chunks
    all_chunks = [c for c in all_chunks if c.page_content.strip()]

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks (by headers)")
    return all_chunks


def enrich_metadata(documents: list[Document]) -> list[Document]:
    """Add useful metadata to documents."""
    for doc in documents:
        source = doc.metadata.get("source", "")

        # Extract filename without extension
        if source:
            filename = source.split("/")[-1].rsplit(".", 1)[0]
            doc.metadata["filename"] = filename

            # Categorize by filename patterns
            filename_lower = filename.lower()
            if "benefit" in filename_lower or "perk" in filename_lower:
                doc.metadata["category"] = "benefits"
            elif "conduct" in filename_lower or "policy" in filename_lower:
                doc.metadata["category"] = "policies"
            elif "title" in filename_lower or "career" in filename_lower:
                doc.metadata["category"] = "career"
            elif "fmla" in filename_lower or "leave" in filename_lower:
                doc.metadata["category"] = "leave"
            elif "device" in filename_lower or "system" in filename_lower:
                doc.metadata["category"] = "it"
            else:
                doc.metadata["category"] = "general"

        # Add content length
        doc.metadata["char_count"] = len(doc.page_content)

    return documents
