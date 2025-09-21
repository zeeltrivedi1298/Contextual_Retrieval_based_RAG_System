# Contextual Retrieval-based RAG (with Sources & Citations)

Build a full Retrieval-Augmented Generation (RAG) workflow that:
- indexes PDFs + JSONL into a vector DB (Chroma),
- answers questions grounded in retrieved context,
- returns **sources** and **verbatim citations** for auditability.

This repo uses **LangChain**, **OpenAI** (GPT-4o-mini + text-embedding-3-small), **Chroma**, and **PyMuPDF**. The notebook is Colab-friendly and also runs locally.

---

## ✨ Features

- **Contextual chunking for PDFs**: each chunk is prefixed with a 3–4 sentence, LLM-generated mini-context to improve retrieval.
- **Hybrid corpus**: loads JSONL (Wiki snippets) + classic DL papers (Attention, ResNet, ViT, CNN).
- **Vector store**: Chroma (cosine distance, persisted to disk).
- **RAG pipelines**:
  - Basic RAG (answer only)
  - RAG with **Sources** (returns the retrieved docs)
  - RAG with **Quoted Citations** (verbatim quotes + highlighted context)
- **Deterministic output** for QA (temperature=0).

---

## 🧱 Architecture (high level)

         ┌───────────┐       ┌─────────────┐       ┌──────────────┐
Data --> │ Loaders   │  -->  │ Chunk+Ctx   │  -->  │ Embeddings    │
(PDF/JSONL PyMuPDF/  │       │ (LLM mini   │       │ (OpenAI       │
 JSONLoader)         │       │ summaries)  │       │ text-emb-3)   │
         └───────────┘       └─────────────┘       └──────┬───────┘
                                                           │
                                                    ┌──────▼───────┐
                                                    │ Chroma (DB)  │
                                                    │ persist_dir  │
                                                    └──────┬───────┘
                                                           │
                                                ┌──────────▼──────────┐
                                                │ Retriever (k=5)     │
                                                └──────────┬──────────┘
                                                           │
                                    ┌───────────────────────┴───────────────────────┐
                                    │     RAG Chains (LangChain + OpenAI LLM)      │
                                    │  - Answer only                                │
                                    │  - Answer + Sources                           │
                                    │  - Answer + Quoted Citations (structured)     │
                                    └───────────────────────────────────────────────┘





---

## 🔑 Requirements

- Python 3.10+
- OpenAI API key with access to `gpt-4o-mini` and `text-embedding-3-small`

Set your API key:

**Colab**
```python
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")


export OPENAI_API_KEY="sk-..."
# Windows (Powershell): $env:OPENAI_API_KEY="sk-..."



python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip

pip install langchain langchain-openai langchain-community langchain-chroma
pip install pymupdf jq

🖥️ Run Locally
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip

pip install langchain langchain-openai langchain-community langchain-chroma
pip install pymupdf jq

⚙️ Key Parameters
Embedding model: text-embedding-3-small
LLM: gpt-4o-mini (temperature=0)
Chunking: RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
Retriever: similarity (cosine), k=5
Chroma: collection_metadata={"hnsw:space": "cosine"}, persist_directory="./my_context_db"

🧪 Example Queries
query = "What is the difference between transformers and vision transformers?"
result = qa_rag_chain.invoke(query)                # Basic RAG
result = rag_chain_w_sources.invoke(query)         # RAG + Sources
result = rag_chain_w_citations.invoke(query)       # RAG + Quoted Citations
