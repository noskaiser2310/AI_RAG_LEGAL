"""AppState: lazy singleton cho corpus, indexes, embedder, reranker, LLM, pipeline.

Khởi tạo một lần duy nhất (lazy), tái sử dụng giữa các request.
"""
import asyncio
import logging
import time
from pathlib import Path

from src.core.config import config
from src.data.loading import load_corpus
from src.embedding.harrier_embedding import HarrierEmbedding
from src.retrieval.indexing import DenseIndex, SparseIndex
from src.reranker.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

_llm_backend = "unknown"
_llm_model = ""


def _create_llm():
    global _llm_backend, _llm_model
    from src.llm.gemini_parallel import GeminiLLM

    keys = [config.GOOGLE_API_KEY] if config.GOOGLE_API_KEY else None
    _llm_backend = "gemini"
    _llm_model = config.GEMINI_MODEL
    return GeminiLLM(model=config.GEMINI_MODEL, keys=keys)


class AppState:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._ready = False
        self.docs = None
        self.dense_idx = None
        self.sparse_idx = None
        self.embedder = None
        self.llm = None
        self.reranker = None
        self.pipeline = None
        self.request_lock = asyncio.Lock()
        self.init_error: str | None = None
        self.init_time = 0.0

    async def ensure_loaded(self):
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            t0 = time.time()
            try:
                logger.info("AppState: loading corpus...")
                self.docs = load_corpus(force_rebuild=False, lazy=True)
                logger.info(f"AppState: corpus {len(self.docs):,} docs")

                logger.info("AppState: loading embedder...")
                self.embedder = HarrierEmbedding()

                logger.info("AppState: loading reranker...")
                self.reranker = CrossEncoderReranker()

                logger.info("AppState: loading indexes...")
                index_dir = Path(config.INDEX_DIR)
                self.dense_idx = DenseIndex()
                self.dense_idx.load(str(index_dir / "dense.index"))
                self.sparse_idx = SparseIndex()
                self.sparse_idx.load(str(index_dir / "sparse"))

                logger.info("AppState: creating LLM...")
                self.llm = _create_llm()

                from src.pipeline.agent import LegalAgent
                from src.retrieval.multi_strategy import MultiStrategyRetriever

                retriever = MultiStrategyRetriever(
                    self.docs, self.dense_idx, self.sparse_idx, self.embedder
                )
                self.pipeline = LegalAgent(
                    llm=self.llm,
                    retriever=retriever,
                    reranker=self.reranker,
                    embedder=self.embedder,
                    docs=self.docs,
                )
                self._ready = True
                self.init_time = time.time() - t0
                logger.info(f"AppState ready in {self.init_time:.1f}s")
            except Exception as e:
                self.init_error = str(e)
                logger.exception("AppState init failed")
                raise

    async def answer(self, query: str, mode: str = "agentic", use_self_correct: bool = True, max_iterations: int = 0):
        await self.ensure_loaded()
        async with self.request_lock:
            return await self.pipeline.answer(
                query,
                max_iterations=max_iterations,
                use_self_correct=use_self_correct,
            )

    async def retrieve_only(self, query: str, top_k: int = 20):
        """Tra cứu + rerank thuần (không gọi LLM), trả chunks đã reconstruct."""
        import numpy as np

        await self.ensure_loaded()
        async with self.request_lock:
            from src.core.base import QueryType

            prefix = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"
            prefixed = f"{prefix}\nTruy vấn: {query}"
            emb_list = await self.embedder.embed([prefixed])
            q_emb = np.array(emb_list[0])
            chunks = self.pipeline.retriever.retrieve_sync(
                query, top_k=200, query_emb=q_emb,
                dense_weight=0.7, sparse_weight=0.3,
            )
            reranked = await self.pipeline.reranker.rerank(query, chunks, top_k=top_k)
            return await self.pipeline._reconstruct_articles(reranked)


state = AppState()
