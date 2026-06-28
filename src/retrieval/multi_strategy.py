import logging

import numpy as np

from src.core.base import BaseRetriever, RetrievedChunk
from src.core.config import config
from src.retrieval.indexing import DenseIndex, SparseIndex, rrf_fusion

logger = logging.getLogger(__name__)

RRF_WEIGHTS_DENSE = 0.8
RRF_WEIGHTS_SPARSE = 0.2


def adaptive_k_by_gap(
    chunks: list[tuple[int, float]],
    buffer: int = 5,
    gap_ratio: float = 0.9,
    min_k: int = 3,
    max_k: int = 200,
) -> list[tuple[int, float]]:
    if len(chunks) <= min_k:
        return chunks
    scores = [s for _, s in chunks]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    cutoff = int(len(gaps) * gap_ratio)
    if cutoff <= 0:
        return chunks[:min_k]
    top_gaps = gaps[:cutoff]
    max_gap_idx = top_gaps.index(max(top_gaps))
    k = max(min_k, min(max_k, max_gap_idx + 1 + buffer))
    return chunks[:k]


class MultiStrategyRetriever(BaseRetriever):
    def __init__(
        self,
        docs: list[dict],
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        embedder=None,
    ):
        self.docs = docs
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedder = embedder
        self.doc_id_to_idx = {i: i for i in range(len(docs))}
        logger.info(f"MultiStrategyRetriever: {len(docs)} chunks")

    async def retrieve(
        self,
        query: str,
        top_k: int = 500,
        query_emb: np.ndarray | None = None,
        dense_weight: float = RRF_WEIGHTS_DENSE,
        sparse_weight: float = RRF_WEIGHTS_SPARSE,
        use_adaptive_k: bool = True,
    ) -> list[RetrievedChunk]:
        return self.retrieve_sync(query, top_k, query_emb, dense_weight, sparse_weight, use_adaptive_k)

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 500,
        query_emb: np.ndarray | None = None,
        dense_weight: float = RRF_WEIGHTS_DENSE,
        sparse_weight: float = RRF_WEIGHTS_SPARSE,
        use_adaptive_k: bool = True,
    ) -> list[RetrievedChunk]:
        rrf_k = min(top_k * 2, len(self.docs))

        dense_results = self._dense_search(query_emb, rrf_k) if query_emb is not None else []
        sparse_results = self._sparse_search(query, rrf_k)

        if use_adaptive_k and dense_results:
            dense_results = adaptive_k_by_gap(dense_results, max_k=rrf_k)
            sparse_results = adaptive_k_by_gap(sparse_results, max_k=rrf_k)

        if dense_results:
            fused = rrf_fusion(
                dense_results, sparse_results,
                weights=[dense_weight, sparse_weight],
                top_k=top_k,
            )
        else:
            fused = sparse_results[:top_k]

        return self._to_chunks(fused)

    def dense_search(self, query_emb: np.ndarray, k: int) -> list[RetrievedChunk]:
        results = self._dense_search(query_emb, k)
        return self._to_chunks(results)

    def sparse_search(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self._sparse_search(query, k)
        return self._to_chunks(results)

    def _dense_search(self, query_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        return self.dense_index.search(query_emb, k)

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.sparse_index.search(query, k)

    def _to_chunks(self, ranked: list[tuple[int, float]]) -> list[RetrievedChunk]:
        chunks = []
        for idx, score in ranked:
            if idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            chunks.append(RetrievedChunk(
                chunk_id=doc.get("chunk_id", str(idx)),
                doc_id=doc.get("doc_id", str(idx)),
                article_id=doc.get("article_id", ""),
                doc_title=doc.get("title", ""),
                content=doc.get("text", ""),
                score=float(score),
                retrieval_score=float(score),
                source="hybrid",
                metadata={
                    "index": idx,
                    "so_ky_hieu": doc.get("so_ky_hieu", ""),
                },
            ))
        return chunks
