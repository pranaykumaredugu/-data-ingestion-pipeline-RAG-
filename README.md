# RAG Pipeline From Scratch

A Retrieval-Augmented Generation (RAG) pipeline built from the ground up — without relying on high-level, one-line abstractions. This project implements each stage of the pipeline manually to demonstrate a clear understanding of how semantic search and document retrieval actually work under the hood.

## What This Project Does

Instead of calling a single black-box function to "build a RAG system," this project breaks the process down into its core components:

**Data Ingestion → Chunking → Embedding → Vector Storage → Retrieval**

Given a collection of text and PDF documents, the system:
1. Loads and parses raw documents
2. Splits them into smaller, semantically meaningful chunks
3. Converts each chunk into a numeric vector (embedding) that captures its meaning
4. Stores these vectors in a persistent vector database
5. Given a natural language query, retrieves the most semantically relevant chunks — even if they don't share exact keywords with the query

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────┐
│  Data Ingestion  │ --> │   Chunking   │ --> │ Embedding  │ --> │Vector Storage│ --> │ Retrieval  │
│ (.txt / .pdf)    │     │ (splitter)   │     │ (vectors)  │     │ (ChromaDB)   │     │  (query)   │
└─────────────────┘     └──────────────┘     └────────────┘     └──────────────┘     └────────────┘
```

| Stage | Purpose | Implementation |
|---|---|---|
| **Data Ingestion** | Load raw `.txt` and `.pdf` files into a unified document format | `TextLoader`, `DirectoryLoader`, `PyMuPDFLoader` (LangChain) |
| **Chunking** | Break large documents into smaller, focused pieces for precise retrieval | `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50) |
| **Embedding** | Convert text chunks into 384-dimensional vectors capturing semantic meaning | Custom `EmbeddingManager` class using `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Storage** | Persistently store chunks + embeddings for fast similarity search | Custom `VectorStore` class using `ChromaDB` (disk-backed, persistent) |
| **Retrieval** | Given a query, find the most semantically similar chunks | Cosine similarity search via ChromaDB's `collection.query()` |

## Why Build This From Scratch?

Most RAG tutorials wrap everything into a single high-level call like `VectorStore.from_documents()`, hiding what's actually happening. This project instead implements each component as its own class with explicit control over:

- How documents are loaded and what metadata is preserved
- How and where documents get split, and why chunk size/overlap matter
- Which embedding model is used and how embeddings are generated
- How data is persisted to disk so it survives across sessions
- How similarity search actually retrieves relevant content

This approach mirrors how production RAG systems are architected — as modular, swappable components rather than a single opaque pipeline.

## Tech Stack

- **[LangChain](https://python.langchain.com/)** / **LangChain Community** — document loaders and text splitters
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** / **PyPDF** — PDF parsing
- **[Sentence Transformers](https://www.sbert.net/)** (`all-MiniLM-L6-v2`) — generating text embeddings
- **[ChromaDB](https://www.trychroma.com/)** — persistent vector database for similarity search
- **NumPy** / **scikit-learn** — vector operations and similarity calculations
- **Python 3.12**

## Project Structure

```
.
├── data/
│   ├── text_files/       # Sample .txt documents
│   ├── pdf_files/         # Sample .pdf documents
│   └── vector_store/      # Persisted ChromaDB vector database
├── notebook/
│   └── document.ipynb     # Main pipeline notebook
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/pranaykumaredugu/-data-ingestion-pipeline-RAG-.git
cd -data-ingestion-pipeline-RAG-
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

Open `notebook/document.ipynb` and run the cells in order. The pipeline will:

1. Load sample text and PDF documents from `data/`
2. Split them into chunks
3. Generate embeddings for each chunk
4. Store the chunks and embeddings in a persistent ChromaDB collection

**Example query:**
```python
query = "What are the types of machine learning?"

query_embedding = embedding_manager.generate_embeddings([query])

results = vectorstore.collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

for doc, metadata, distance in zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
):
    print(f"Source: {metadata.get('source')} | Distance: {distance:.4f}")
    print(doc)
```

This returns the top 3 chunks most semantically relevant to the query — even if the exact wording differs from the source documents.

## Future Improvements

- [ ] Integrate an LLM (e.g., Claude, GPT) to generate natural language answers from retrieved chunks, completing the full RAG loop
- [ ] Add a simple chat interface (Streamlit / Gradio) for interactive querying
- [ ] Add evaluation examples comparing retrieved chunks against expected results
- [ ] Support additional document types (Word, HTML, Markdown)
- [ ] Add deduplication logic to prevent duplicate chunks on repeated ingestion runs

## License

This project is open source and available for learning and portfolio purposes.
