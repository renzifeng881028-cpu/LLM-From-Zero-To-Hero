# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese legal RAG (retrieval-augmented generation) Q&A web app covering the Civil Code of the PRC (中华人民共和国民法典, 1260 articles). Single-file Streamlit app built on LlamaIndex + ChromaDB + OpenAI models. Part of the larger "LLM-From-Zero-To-Hero" learning repo; work stays inside this `minfa` folder.

## Commands

```bash
# Dependencies (script header lists the core three; dotenv and the chroma
# vector-store integration are also imported)
pip install streamlit llama-index chromadb llama-index-vector-stores-chroma python-dotenv

# Requires a .env file in this directory with:
#   OPENAI_API_KEY=...
#   OPENAI_BASE_URL=...      (used as api_base for both LLM and embeddings)

# Run the app (index is built/reused automatically on startup)
streamlit run rag_law_minfa_web_gpt-4o-mini.py
```

There is no build step, no test suite, and no lint configuration. The script also has a `__main__` guard, but it is a Streamlit app and should be launched via `streamlit run`.

## Architecture

The entire pipeline lives in `rag_law_minfa_web_gpt-4o-mini.py`:

1. **Data loading** — every `*.json` in `data/` must be a JSON *list of flat dicts* mapping `"<法律名> 第N条"` (law name + article) → article text. Validation raises on any non-string value. `data/data_minfa.json` currently holds all 1260 civil-code articles as one list item.
2. **Node creation** (`create_nodes`) — one LlamaIndex `TextNode` per article; node ID is `"{source_file}::{full_title}"`, and metadata splits the title into `law_name` / `article` on the first space. This metadata contract is what the UI renders as "支持依据" (sources) after each answer — keep these keys stable.
3. **Vector store** (`init_vector_store`) — Chroma persistent client in `chroma_db/`, collection `chinese_labor_laws` (cosine distance), wrapped for LlamaIndex; LlamaIndex `StorageContext` (docstore etc.) persists separately in `storage/`.
4. **Query** — `VectorStoreIndex.as_query_engine` with `TOP_K = 3` and a custom ChatML-style (`<|im_start|>` tags) QA template defined in `QA_TEMPLATE`; models are set globally via LlamaIndex `Settings` (`gpt-4o-mini` LLM at temperature 0, `text-embedding-3-small` embeddings). `init_models()` makes a probe embedding call on every startup to verify the embedding endpoint.

### Index-rebuild gating (important)

The index is rebuilt **only if the `chroma_db/` directory does not exist** (checked in `main()`) AND the Chroma collection is empty (checked in `init_vector_store()`). Consequences:

- To force re-ingestion after changing `data/`, delete **both** `chroma_db/` and `storage/` before restarting.
- Deleting only `storage/` leaves the app loading from the existing Chroma collection; deleting only `chroma_db/` but having the recreated collection still empty while `nodes` was skipped yields an empty index. Both directories must be in sync.

### Known inconsistencies (inherited, not bugs to "fix" silently)

- The Chroma collection is named `chinese_labor_laws` and a commented-out UI line references labor law (劳动法), but the actual data is civil law (民法). The naming is a leftover from an earlier labor-law variant of this project.
- `QA_TEMPLATE` uses ChatML markup intended for local models; it is passed as a raw prompt template to the OpenAI model.

### UI notes

Streamlit `st.session_state` holds the question box under key `input_question`; the "清除查询结果" button clears the whole `session_state` via an `st.fragment` callback (clearing only the input key raises an error — the code comments document this).
