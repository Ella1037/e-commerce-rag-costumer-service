# app/main.py
import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.rag import CHAINS
from app.cache import query_cache

app = FastAPI(title="RAG Inference Server")

class QueryRequest(BaseModel):
    question: str
    method: Literal["baseline", "reranker", "hyde"] = "baseline"

class QueryResponse(BaseModel):
    answer: str
    method: str
    latency_ms: float
    cache_hit: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/cache/stats")
def cache_stats():
    return query_cache.stats

@app.delete("/cache")
def clear_cache():
    query_cache._cache.clear()
    return {"status": "cleared"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    chain = CHAINS.get(req.method)
    if not chain:
        raise HTTPException(status_code=400, detail="Unknown method")

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    # ── Cache lookup ──────────────────────────────────────────
    cached = query_cache.get(req.question, req.method)
    if cached:
        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResponse(
            answer=cached,
            method=req.method,
            latency_ms=latency_ms,
            cache_hit=True,
        )

    # ── Cache miss: run full pipeline ─────────────────────────
    loop   = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, chain.invoke, req.question)
    query_cache.set(req.question, req.method, answer)

    latency_ms = (time.perf_counter() - t0) * 1000
    return QueryResponse(
        answer=answer,
        method=req.method,
        latency_ms=latency_ms,
        cache_hit=False,
    )