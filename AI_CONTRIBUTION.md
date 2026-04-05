# AI Contribution Notes

This project was built iteratively with AI assistance across four steps.
Below is an honest account of how AI was used, what decisions I made, and
where the iterative process surfaced problems that forced design changes.

## Step 1 — Ingestion

- Drafted the ingestion script with AI help: markdown loader, paragraph-aware
  chunker (1000 chars, 200 overlap), ChromaDB `PersistentClient` with cosine
  similarity.
- **My decision:** paragraph-boundary splitting instead of naive character
  splits. AI initially suggested fixed-window chunking; I pushed for paragraph
  splits after inspecting sample chunks that awkwardly cut mid-sentence.
- **Outcome:** 802 chunks from ~50 markdown files. Verified chunk quality by
  eyeballing samples.

## Step 2 — Retrieval

- Used AI to scaffold the query-and-inspect loop, but I did the threshold
  tuning empirically.
- **Tuning journey:** started at `0.35` (too aggressive — filtered valid
  results for "handbook first"). Tried `0.65` (too loose — pulled in unrelated
  chunks for the stock price query). Settled on `0.55` after testing against
  the golden dataset.
- **My decision:** rejected AI's suggestion to add a BM25 hybrid pass. Kept
  the retriever simple since cosine alone hit the required recall on the
  golden set.

## Step 3 — Generation

- System prompt drafted with AI, then hardened manually after observing
  failure modes:
  - "What is GitLab's hybrid work policy?" → model tried to invent a hybrid
    policy from remote-work context. Fixed by adding explicit "don't infer
    beyond context" language.
  - "What is GitLab's stock price?" → model leaked outside knowledge. Fixed
    by adding a concrete stock-price example to the prompt.
  - "What is the capital of France?" → clean refusal from the first draft.
- **My decision:** kept citations subtle (italic single line) instead of the
  bold block AI suggested. The assignment requires citations but they
  shouldn't clutter the answer.

## Step 4 — Chat memory + UI

- Gradio `ChatInterface` wired up with AI help.
- **My decision:** the "vague follow-up enrichment" (if user message <30
  chars, prepend previous user message before retrieval) is mine. AI
  originally passed full history to the retriever, which polluted the query
  with keywords from unrelated turns.
- History formatting handles both tuple and dict formats because the Gradio
  version in use returns tuples but future versions emit dicts.

## Evaluation

- `golden_dataset.json` has 9 Q&A pairs (7 positive, 2 negative). Q2 was
  rephrased after an earlier version had a retrieval miss.
- `test_results.json` produced by `evaluate.py` with a 5-second sleep between
  calls to respect Gemini free-tier rate limits.
- **What I did not do:** automated LLM-as-judge scoring. The golden set is
  small enough to review by hand, and adding judge prompts would introduce
  another model to tune.

## What AI was good at

- Boilerplate (Gradio setup, ChromaDB init, LiteLLM call pattern)
- Suggesting reasonable defaults (chunk size, top-k)
- Rapid prototyping of the system prompt

## What AI got wrong (and I had to fix)

- Initial chunker was character-based, not paragraph-aware
- First system prompt allowed outside knowledge ("GTLB trades at…" hallucinations)
- Proposed over-engineered hybrid retrieval before the simple retriever had been measured
- Passed full chat history to the retriever, polluting queries
