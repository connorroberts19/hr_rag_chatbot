"""RAG chain implementation for HR document Q&A."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.generation.llm import get_llm
from src.generation.prompts import RAG_PROMPT
from src.retrieval.vector_store import get_retriever


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("filename", "Unknown")
        category = doc.metadata.get("category", "general")
        formatted.append(f"[Source: {source} | Category: {category}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():
    """
    Create the RAG chain for answering HR questions.

    Pipeline:
    1. Retrieve relevant documents from vector store
    2. Format documents into context
    3. Generate answer using LLM with context
    """
    retriever = get_retriever()
    llm = get_llm()

    # Build the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> str:
    """
    Ask a question and get an answer from the HR documents.

    Args:
        question: The HR-related question to answer

    Returns:
        The generated answer based on retrieved documents
    """
    chain = create_rag_chain()
    return chain.invoke(question)


def ask_with_sources(question: str) -> dict:
    """
    Ask a question and get an answer with source documents.

    Args:
        question: The HR-related question to answer

    Returns:
        Dict with 'answer' and 'sources' keys
    """
    retriever = get_retriever()
    llm = get_llm()

    # Get documents first
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Generate answer
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Extract source info
    sources = [
        {
            "filename": doc.metadata.get("filename", "Unknown"),
            "category": doc.metadata.get("category", "general"),
            "excerpt": doc.page_content[:200] + "...",
        }
        for doc in docs
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
