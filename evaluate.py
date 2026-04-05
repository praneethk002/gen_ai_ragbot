"""
evaluate.py - Run the chatbot against the golden dataset and save results.

Standalone replacement for step3_generation.ipynb cell 3.6. Uses a longer
sleep between calls (5 seconds) to stay under Gemini free-tier rate limits.

Usage:
    export GEMINI_API_KEY="your-key"
    python evaluate.py
"""

import os
import json
import time

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
import litellm

# ── Config (must match app.py) ─────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gitlab_handbook"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.55
MODEL = "gemini/gemma-3-27b-it"
SLEEP_BETWEEN_CALLS = 5  # seconds — conservative for Gemini free tier

SYSTEM_PROMPT = """You are a helpful, conversational assistant that answers questions about the GitLab Handbook.

RULES:
1. Answer ONLY based on the provided context from the GitLab Handbook. Do not use any outside knowledge.
2. If the context does not contain the specific information needed to answer the question, say so naturally. For example, if someone asks about a stock price but the context only discusses GitLab being a public company, explain what the handbook does cover and what it doesn't.
3. Do not speculate or infer beyond what is explicitly stated in the context.
4. Be concise, direct, and conversational. Write like a knowledgeable colleague, not a search engine.
5. Only mention source filenames if the user asks where the information comes from, or if citing the source adds clarity. Do not append a sources list to every answer.
6. If no context is provided at all, tell the user you couldn't find anything relevant in the handbook.
7. Use the conversation history to understand follow-up questions. If the user says "why?" or "tell me more", relate it back to the previous topic."""


def retrieve(collection, query: str):
    results = collection.query(query_texts=[query], n_results=TOP_K)
    context_parts = []
    sources = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        if distance <= RELEVANCE_THRESHOLD:
            context_parts.append(f"[Source: {source}]\n{doc}")
            if source not in sources:
                sources.append(source)
    context = "\n\n---\n\n".join(context_parts) if context_parts else ""
    return context, sources


def build_prompt(question: str, context: str) -> str:
    if context:
        return (
            f"CONTEXT FROM GITLAB HANDBOOK:\n{context}\n\n"
            f"CONVERSATION HISTORY:\n(No previous conversation)\n\n"
            f"USER QUESTION: {question}\n\n"
            "Answer the question using ONLY the context above. "
            "If the context doesn't contain the specific answer, say so."
        )
    return (
        "No relevant context was found in the GitLab Handbook for this question.\n\n"
        "CONVERSATION HISTORY:\n(No previous conversation)\n\n"
        f"USER QUESTION: {question}\n\n"
        "Tell the user that you couldn't find relevant information in the GitLab Handbook "
        "to answer their question."
    )


def ask(collection, question: str):
    context, sources = retrieve(collection, question)
    prompt = build_prompt(question, context)
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error calling LLM: {str(e)}"

    if sources:
        citation_block = "\n\n---\n**Sources:**\n" + "\n".join(f"- `{s}`" for s in sources)
        answer += citation_block

    return answer, sources, bool(context)


def main():
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Loaded ChromaDB: {collection.count()} chunks")

    with open("golden_dataset.json", "r") as f:
        golden = json.load(f)

    results = []
    for item in golden:
        question = item["question"]
        print(f"[Q{item['id']}] {question}")
        answer, sources, context_found = ask(collection, question)
        results.append({
            "id": item["id"],
            "question": question,
            "type": item["type"],
            "expected_answer": item["expected_answer"],
            "chatbot_answer": answer,
            "retrieved_sources": sources,
            "context_found": context_found,
        })
        print(f"  Sources: {sources}")
        print(f"  Answer preview: {answer[:150]}...\n")
        time.sleep(SLEEP_BETWEEN_CALLS)

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Saved {len(results)} results to test_results.json")


if __name__ == "__main__":
    main()
