---
title: GitLab Handbook RAG Chatbot
emoji: 📘
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---



# GitLab Handbook RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the
GitLab Handbook. It retrieves relevant sections of the handbook with ChromaDB
and generates grounded answers with Gemma 3 via LiteLLM.

## Stack

- **Vector DB:** ChromaDB (`PersistentClient`, cosine similarity)
- **Embeddings:** ChromaDB default (`all-MiniLM-L6-v2`)
- **LLM:** `gemini/gemma-3-27b-it` via LiteLLM
- **UI:** Gradio `ChatInterface`

## Pipeline

1. **Ingestion** (`ingest.py`): loads ~50 markdown files from `datasets/`,
   produces 802 paragraph-aware chunks (1000 chars, 200 overlap), embeds them
   into ChromaDB.
2. **Retrieval** (`app.py`): top-5 cosine nearest neighbours, filtered by a
   relevance threshold of `0.55` (tuned empirically — `0.35` was too strict,
   `0.65` let irrelevant chunks through).
3. **Generation** (`app.py`): system prompt forces answers to stay grounded in
   the retrieved context; out-of-scope questions are refused naturally.
4. **Chat memory**: vague follow-ups (<30 chars) are enriched with the previous
   user message before retrieval.

## Running locally

```bash
pip install -r requirements.txt

# Build the vector store (only needed once — produces chroma_db/)
python ingest.py

# Run the chatbot
export GEMINI_API_KEY="your-key"
python app.py
```

Then open http://localhost:7860.

## Running the evaluation

```bash
export GEMINI_API_KEY="your-key"
python evaluate.py
```

This queries the chatbot against `golden_dataset.json` (9 Q&A pairs: 7 positive,
2 negative) with a 5-second delay between calls to respect Gemini free-tier
rate limits, and saves results to `test_results.json`.

## Files

| File | Description |
| ---- | ----------- |
| `app.py` | Gradio chatbot UI + RAG pipeline |
| `ingest.py` | Builds ChromaDB from `datasets/` |
| `evaluate.py` | Runs golden dataset evaluation |
| `golden_dataset.json` | 9 hand-written Q&A pairs for evaluation |
| `test_results.json` | Results of the golden dataset evaluation |
| `chroma_db/` | Persisted vector store (802 chunks) |
| `AI_CONTRIBUTION.md` | Notes on how AI was used during development |
| `datasets/` | Source markdown files from the GitLab Handbook |

## Deployment (Hugging Face Spaces)

The `chroma_db/` folder is committed to the repo, so Spaces can load it
directly at startup with no ingestion step. Add `GEMINI_API_KEY` as a repo
secret before launching the Space.
