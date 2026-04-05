"""
ingest.py - Build the ChromaDB vector store from the GitLab Handbook dataset.

Replicates step1_ingestion.ipynb so the chroma_db/ folder can be rebuilt from
scratch if needed. Run once before the first app launch.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DATASET_DIR = "datasets"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gitlab_handbook"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_markdown_files(directory: str) -> list[dict]:
    documents = []
    md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    for filepath in sorted(md_files):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if content:
            rel_path = os.path.relpath(filepath, directory)
            documents.append({"content": content, "source": rel_path})
    return documents


def chunk_document(text: str, source: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": source,
                "chunk_id": f"{source}::chunk_{chunk_index}",
            })
            chunk_index += 1
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "source": source,
            "chunk_id": f"{source}::chunk_{chunk_index}",
        })
    return chunks


def main():
    documents = load_markdown_files(DATASET_DIR)
    print(f"Loaded {len(documents)} markdown files")

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc["content"], doc["source"]))
    print(f"Produced {len(all_chunks)} chunks")

    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
        print(f"Batch {i // batch_size + 1}: added {len(batch)} chunks")

    print(f"Done. Total chunks in collection: {collection.count()}")


if __name__ == "__main__":
    main()
