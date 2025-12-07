# HR Document RAG Chatbot

An AI-powered chatbot that answers HR-related questions by retrieving information from HR policy documents using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)

## Features

- **Semantic Search** - Find relevant information across HR documents using vector embeddings
- **Natural Language Q&A** - Ask questions in plain English and get accurate answers
- **Source Citations** - Every answer includes references to source documents
- **Fully Local** - Runs entirely on your machine with Ollama (no API costs)
- **Fast Retrieval** - ChromaDB vector store for quick document lookup

## Demo

Ask questions like:
- "What is the vacation policy?"
- "How does profit sharing work?"
- "What benefits does the company offer?"
- "How do I request a new laptop?"

## Project Structure

```
hr_rag_chatbot/
├── src/
│   ├── ingestion/      # Document loading and chunking
│   ├── retrieval/      # Vector store and semantic search
│   ├── generation/     # LLM and RAG chain
│   └── app.py          # Streamlit chat interface
├── data/
│   ├── raw/            # Source HR documents (Markdown/TXT)
│   └── processed/      # Chunked documents (JSON)
├── chroma_db/          # Persisted vector store
└── requirements.txt
```

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/connorroberts19/hr_rag_chatbot.git
cd hr_rag_chatbot
python -m venv hr_venv
source hr_venv/bin/activate  # On Windows: hr_venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama (local LLM)
```bash
# Download from https://ollama.ai
# Then pull a model:
ollama pull phi3
```

### 4. Build the document index
```bash
python -m src.retrieval.indexer
```

### 5. Run the application
```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│  Retriever  │────▶│  Top K Docs │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│     LLM     │◀────│   Context   │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Ingestion** - Documents are loaded, chunked by headers, and enriched with metadata
2. **Embedding** - Chunks are converted to vectors using sentence-transformers
3. **Indexing** - Vectors are stored in ChromaDB for fast similarity search
4. **Retrieval** - User questions are embedded and matched against document chunks
5. **Generation** - Retrieved context is passed to Ollama LLM to generate answers

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB |
| **LLM** | Ollama (phi3) |
| **Framework** | LangChain |
| **Frontend** | Streamlit |

## Configuration

Edit `.env` to customize:

```bash
# Change the Ollama model
OLLAMA_MODEL=phi3  # or llama3.2, mistral, etc.
```

## Adding Your Own Documents

1. Place Markdown or text files in `data/raw/`
2. Rebuild the index:
   ```bash
   python -m src.retrieval.indexer --force
   ```

## License

MIT
