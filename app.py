"""
app.py - GitLab Handbook RAG Chatbot

Gradio chat interface over ChromaDB + LiteLLM (Gemma 3).
Deploy to Hugging Face Spaces with the chroma_db/ folder included.

Local usage:
    export GEMINI_API_KEY="your-key"
    python app.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
import litellm
import gradio as gr

# ── Config ─────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gitlab_handbook"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.55
MODEL = "gemini/gemma-3-27b-it"

SYSTEM_PROMPT = """You are a helpful, conversational assistant that answers questions about the GitLab Handbook.

RULES:
1. Answer ONLY based on the provided context from the GitLab Handbook. Do not use any outside knowledge.
2. If the context does not contain the specific information needed to answer the question, say so naturally. For example, if someone asks about a stock price but the context only discusses GitLab being a public company, explain what the handbook does cover and what it doesn't.
3. Do not speculate or infer beyond what is explicitly stated in the context.
4. Be concise, direct, and conversational. Write like a knowledgeable colleague, not a search engine.
5. Only mention source filenames if the user asks where the information comes from, or if citing the source adds clarity. Do not append a sources list to every answer.
6. If no context is provided at all, tell the user you couldn't find anything relevant in the handbook.
7. Use the conversation history to understand follow-up questions. If the user says "why?" or "tell me more", relate it back to the previous topic."""

# ── Load ChromaDB ──────────────────────────────────────────────────────
ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)
print(f"Loaded ChromaDB: {collection.count()} chunks")


# ── Retrieval ──────────────────────────────────────────────────────────
def retrieve(query: str) -> tuple[str, list[str]]:
    """Query ChromaDB, filter by threshold, return context + sources."""
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


# ── Chat history formatting ────────────────────────────────────────────
def format_history(history: list) -> str:
    """Convert Gradio history (list of tuples) to string for prompt."""
    if not history:
        return "(No previous conversation)"
    lines = []
    for item in history[-6:]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lines.append(f"User: {item[0]}")
            if item[1]:
                lines.append(f"Assistant: {item[1]}")
        elif isinstance(item, dict):
            role = "User" if item["role"] == "user" else "Assistant"
            lines.append(f"{role}: {item['content']}")
    return "\n".join(lines) if lines else "(No previous conversation)"


# ── Main chat function ─────────────────────────────────────────────────
def enrich_query(user_message: str, history: list) -> str:
    """
    If the user's message is short/vague (like 'why?' or 'tell me more'),
    prepend context from the last exchange so retrieval has something to work with.
    """
    vague_threshold = 30  # characters
    if len(user_message.strip()) < vague_threshold and history:
        # Grab the last user message from history for context
        last_user_msg = None
        for item in reversed(history):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                last_user_msg = item[0]
                break
            elif isinstance(item, dict) and item.get("role") == "user":
                last_user_msg = item["content"]
                break
        if last_user_msg:
            return f"{last_user_msg} {user_message}"
    return user_message


def chat(user_message: str, history: list) -> str:
    if not user_message.strip():
        return "Please ask a question about the GitLab Handbook."

    # 1. Enrich vague follow-ups with previous context, then retrieve
    search_query = enrich_query(user_message, history)
    context, sources = retrieve(search_query)

    # 2. Build prompt
    history_str = format_history(history)

    if context:
        prompt = f"""CONTEXT FROM GITLAB HANDBOOK:
{context}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_message}

Answer the question using ONLY the context above. If the context doesn't contain the specific answer, say so."""
    else:
        prompt = f"""No relevant context was found in the GitLab Handbook for this question.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_message}

Tell the user that you couldn't find relevant information in the GitLab Handbook to answer their question."""

    # 3. Call LLM
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
        answer = f"Sorry, there was an error generating a response: {str(e)}"

    # 4. Append source citations (required by assignment spec)
    # Kept subtle so the answer reads naturally
    if sources:
        source_list = ", ".join(sources)
        answer += f"\n\n*📄 Sources: {source_list}*"

    return answer


# ── Gradio UI ──────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="GitLab Handbook RAG Chatbot",
    description=(
        "Ask me anything about the GitLab Handbook: engineering practices, "
        "remote work policies, company culture, and more. "
        "I retrieve the relevant sections and answer based on the actual handbook content."
    ),
    examples=[
        "What does handbook first mean?",
        "What is GitLab's approach to remote work?",
        "When should I escalate a handbook issue?",
        "What is the capital of France?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
