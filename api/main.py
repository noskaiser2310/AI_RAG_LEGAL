"""LegalRAG Product API.

Endpoints:
- GET  /api/health         kiểm tra trạng thái + model
- GET  /api/stats          thống kê corpus
- POST /api/chat           trả lời câu hỏi pháp lý (JSON)
- POST /api/chat/stream    trả lời streaming (SSE: stage + delta + done)
- POST /api/retrieve       tra cứu + rerank, trả chunks (không sinh câu trả lời)
"""
import asyncio
import json
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from src.core.config import config
from src.retrieval.text_processor import extract_article_references

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CorpusStats,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkOut,
)
from api.state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="LegalRAG API",
    description="Trợ lý pháp lý AI cho doanh nghiệp SME Việt Nam - RAG + Agentic retrieval",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chunk_out(c) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        article_id=c.article_id,
        doc_title=c.doc_title,
        content=c.content,
        score=c.score,
        retrieval_score=c.retrieval_score,
        rerank_score=c.rerank_score,
        source=c.source,
        metadata=c.metadata,
    )


def _result_to_response(result) -> ChatResponse:
    total = (
        result.retrieval_time + result.rerank_time + result.generation_time
    )
    return ChatResponse(
        query=result.query,
        query_type=result.query_type,
        answer=result.final_answer or result.answer,
        citations=result.citations,
        relevant_docs=result.relevant_docs,
        relevant_articles=result.relevant_articles,
        expanded_queries=result.expanded_queries,
        confidence=result.confidence,
        retrieval_time=result.retrieval_time,
        rerank_time=result.rerank_time,
        generation_time=result.generation_time,
        total_time=total,
        num_correction_rounds=result.num_correction_rounds,
        chunks=[_chunk_out(c) for c in result.chunks],
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    status = "loading"
    if state._ready:
        status = "ready"
    elif state.init_error:
        status = "error"
    info = {
        "status": status,
        "app": config.APP_NAME,
        "corpus_docs": len(state.docs) if state.docs is not None else None,
        "dense_index_vectors": state.dense_idx.index.ntotal if state.dense_idx else None,
        "sparse_index_docs": len(state.sparse_idx.doc_ids) if state.sparse_idx else None,
        "llm_backend": "gemini",
        "llm_model": config.GEMINI_MODEL,
        "device": config.EMBEDDING_DEVICE or config.DEVICE,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
    }
    if state.init_error:
        info["detail"] = state.init_error
    return info


@app.get("/api/stats", response_model=CorpusStats)
async def stats():
    await state.ensure_loaded()
    sources: dict[str, int] = {}
    with_articles = 0
    with_so_ky_hieu = 0
    top_law_counts: dict[str, int] = {}
    for doc in state.docs:
        src = doc.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        if doc.get("article_id"):
            with_articles += 1
        if doc.get("so_ky_hieu"):
            with_so_ky_hieu += 1
            key = doc["so_ky_hieu"].split("|")[0]
            top_law_counts[key] = top_law_counts.get(key, 0) + 1
    top_laws = [
        {"so_ky_hieu": k, "chunks": v}
        for k, v in sorted(top_law_counts.items(), key=lambda x: -x[1])[:15]
    ]
    return CorpusStats(
        total_docs=len(state.docs),
        sources=sources,
        with_articles=with_articles,
        with_so_ky_hieu=with_so_ky_hieu,
        top_laws=top_laws,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()
    logger.info(f"Chat: {req.query[:100]} (mode={req.mode})")
    result = await state.answer(
        req.query,
        mode=req.mode,
        use_self_correct=req.use_self_correct,
        max_iterations=req.max_iterations,
    )
    resp = _result_to_response(result)
    logger.info(
        f"Chat done in {time.time()-t0:.1f}s: type={resp.query_type}, "
        f"chunks={len(resp.chunks)}, citations={len(resp.citations)}"
    )
    return resp


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_stream():
        async def send(event: str, data: dict):
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            yield "event: stage\ndata: {\"name\": \"starting\", \"message\": \"Khởi tạo hệ thống...\"}\n\n"
            await state.ensure_loaded()

            t0 = time.time()
            result = await state.answer(
                req.query,
                mode=req.mode,
                use_self_correct=req.use_self_correct,
                max_iterations=req.max_iterations,
            )
            resp = _result_to_response(result)
            resp.total_time = time.time() - t0
            yield "event: done\ndata: " + resp.model_dump_json() + "\n\n"
        except Exception as e:
            logger.exception("Chat stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)[:500]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    """Chỉ tra cứu + rerank, không gọi LLM sinh câu trả lời."""
    chunks = await state.retrieve_only(req.query, top_k=req.top_k)
    return RetrievalResponse(
        query=req.query,
        query_type="retrieve",
        chunks=[_chunk_out(c) for c in chunks[: req.top_k]],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)[:500]})
