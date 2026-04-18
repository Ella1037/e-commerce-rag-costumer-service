# E-Commerce RAG Customer Service Chatbot

A production-style Retrieval-Augmented Generation (RAG) system for e-commerce customer service, comparing three retrieval strategies with an in-memory cache layer and quantitative benchmarking.

<!-- ![demo](demo.gif) -->

---

## Architecture

```
User Query
    │
    ▼
FastAPI /query
    │
    ├── Cache hit (~0.015ms) ──────────────────────────► Response
    │
    └── Cache miss
            │
            ▼
    ┌─────────────────────────────────┐
    │  Retrieval Strategy             │
    │                                 │
    │  1  Bi-encoder baseline         │
    │  2  + Cross-encoder reranker    │
    │  3  HyDE query expansion        │
    └─────────────────────────────────┘
            │
            ▼
       Top-3 Documents
            │
            ▼
    LLM (llama-3.1-8b-instant via Groq)
            │
            ▼
    Cache set → Response
```

## Retrieval Methods

| Method         | How it works                                          | Cold latency (mean) |
| -------------- | ----------------------------------------------------- | ------------------- |
| **Bi-encoder** | Embed query and docs independently, cosine similarity | ~340ms              |
| **Reranker**   | Bi-encoder top-15 → Cross-encoder rerank → top-3      | ~447ms              |
| **HyDE**       | LLM generates hypothetical answer → embed → search    | ~676ms              |

Cache warm latency (LRU in-memory, TTL=1h): **~0.015ms** across all methods.

## Tech Stack

| Component     | Tool                                       |
| ------------- | ------------------------------------------ |
| LLM           | `llama-3.1-8b-instant` via Groq            |
| Embeddings    | `thenlper/gte-small` (ONNX int8 quantized) |
| Reranker      | `cross-encoder/ms-marco-MiniLM-L-6-v2`     |
| Vector Store  | FAISS                                      |
| Framework     | LangChain (LCEL)                           |
| API Server    | FastAPI                                    |
| Cache         | LRU in-memory (500 entries, 1h TTL)        |
| UI (notebook) | Gradio                                     |

## Project Structure

```
app/
  main.py        # FastAPI server
  rag.py         # RAG chains (baseline / reranker / HyDE)
  embeddings.py  # ONNX-quantized embedding wrapper
  cache.py       # LRU query cache
  benchmark.py   # Latency benchmarking script
scripts/
  load_test.py   # Locust load test
models/          # ONNX model weights (not tracked in git)
RAG_CustomerService_Chatbot.ipynb  # Original Gradio notebook
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Download ONNX models (required before running the server)
python scripts/export_onnx.py   # or follow notebook instructions

# 4. Start the API server
uvicorn app.main:app --reload
```

## API Usage

```bash
# Query with a specific retrieval method
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I return an item?", "method": "reranker"}'

# Check cache stats
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

Response:

```json
{
  "answer": "You can return items within 15 days of delivery...",
  "method": "reranker",
  "latency_ms": 423.1,
  "cache_hit": false
}
```

## Notebook Demo

For the original Gradio UI, open `RAG_CustomerService_Chatbot.ipynb` in Google Colab:

1. Add `GROQ_API_KEY` to Colab Secrets (left sidebar → key icon)
2. Run all cells top to bottom
3. A public Gradio link will be generated valid for 72 hours
