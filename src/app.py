"""Streamlit chat interface for the HR RAG Chatbot."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.generation import ask_with_sources
from src.retrieval import get_index_stats

# Page config
st.set_page_config(
    page_title="HR Assistant",
    page_icon="👔",
    layout="centered",
)

# Header
st.title("👔 HR Assistant")
st.markdown("Ask questions about company policies, benefits, and more.")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This assistant answers questions based on company HR documents.

        **Topics covered:**
        - Benefits & perks
        - Vacation & leave policies
        - Career development
        - Company policies
        - IT & devices
        """
    )

    # Show index stats
    st.divider()
    st.subheader("📊 Index Stats")
    stats = get_index_stats()
    if "error" not in stats:
        st.metric("Documents indexed", stats["document_count"])
    else:
        st.warning("Index not available")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['filename']}** ({source['category']})")
                    st.caption(source["excerpt"])

# Chat input
if prompt := st.chat_input("Ask an HR question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents..."):
            try:
                result = ask_with_sources(prompt)
                response = result["answer"]
                sources = result["sources"]

                st.markdown(response)

                # Show sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"**{source['filename']}** ({source['category']})")
                            st.caption(source["excerpt"])

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                })
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
