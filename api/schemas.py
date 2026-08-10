"""Pydantic schemas cho LegalRAG API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Câu hỏi pháp lý")
    mode: str = Field("agentic", description="'pipeline' hoặc 'agentic'")
    use_self_correct: bool = True
    max_iterations: int = Field(0, ge=0, le=5)


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(20, ge=1, le=100)


class RetrievedChunkOut(BaseModel):
    chunk_id: str = ""
    doc_id: str = ""
    article_id: str = ""
    doc_title: str = ""
    content: str = ""
    score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    query: str
    query_type: str = "unknown"
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    relevant_docs: list[str] = Field(default_factory=list)
    relevant_articles: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    num_correction_rounds: int = 0
    chunks: list[RetrievedChunkOut] = Field(default_factory=list)


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    query_type: str = "unknown"
    expanded_queries: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    corpus_docs: Optional[int] = None
    dense_index_vectors: Optional[int] = None
    sparse_index_docs: Optional[int] = None
    llm_backend: str = "unknown"
    llm_model: str = ""
    device: str = ""
    embedding_model: str = ""
    reranker_model: str = ""


class CorpusStats(BaseModel):
    total_docs: int
    sources: dict[str, int]
    with_articles: int
    with_so_ky_hieu: int
    top_laws: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
