import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from src.core.config import config
from src.retrieval.text_processor import prepare_for_bm25

logger = logging.getLogger(__name__)


class DenseIndex:
    def __init__(self, dimension: int | None = None):
        self.dimension = dimension or config.EMBEDDING_DIM
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(self.index)

    def add(self, embeddings: np.ndarray, ids: np.ndarray):
        self.index.add_with_ids(embeddings.astype(np.float32), ids.astype(np.int64))
        logger.info(f"Dense index: {self.index.ntotal} vectors, dim={self.dimension}")

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        scores, indices = self.index.search(query_emb.astype(np.float32), k)
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx >= 0:
                results.append((idx, float(scores[0][i])))
        return results

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info(f"Dense index saved to {path}")

    def load(self, path: str | Path):
        self.index = faiss.read_index(str(path))
        logger.info(f"Dense index loaded: {self.index.ntotal} vectors")


class SparseIndex:
    def __init__(self, use_vn_segmentation: bool = True):
        self.bm25 = None
        self.corpus: list[str] = []
        self.doc_ids: list[int] = []
        self.use_vn_segmentation = use_vn_segmentation
        self._engine = "bm25s"  # "bm25s" | "rank_bm25"

    def _tokenize(self, text: str) -> list[str]:
        norm = prepare_for_bm25(text)
        if self.use_vn_segmentation:
            try:
                from src.retrieval.segmentation import tokenize_vietnamese
                return tokenize_vietnamese(norm)
            except Exception:
                pass
        return norm.split()

    def build(self, texts: list[str], doc_ids: list[int]):
        import bm25s as bm25s_lib
        logger.info(f"Building BM25 index with bm25s ({len(texts)} docs)...")
        tokenized = bm25s_lib.tokenize(texts, show_progress=True)
        self.bm25 = bm25s_lib.BM25(k1=config.BM25_K1, b=config.BM25_B)
        self.bm25.index(tokenized)
        self.corpus = texts
        self.doc_ids = doc_ids
        self._engine = "bm25s"
        vocab_size = len(self.bm25.vocab_dict) if hasattr(self.bm25, 'vocab_dict') else 0
        logger.info(f"Sparse index built: {len(texts)} docs, {vocab_size:,} terms")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        import bm25s as bm25s_lib
        tokenized = bm25s_lib.tokenize([query], show_progress=False)
        # bm25s.retrieve trả về (indices, scores) — KHÔNG phải (scores, indices)
        indices, scores = self.bm25.retrieve(tokenized, k=top_k)
        results = []
        for idx in range(len(indices[0])):
            doc_pos = int(indices[0][idx])
            if doc_pos >= 0 and doc_pos < len(self.doc_ids):
                doc_id = self.doc_ids[doc_pos]
                results.append((doc_id, float(scores[0][idx])))
        return results

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(path))
        meta = {
            "doc_ids": self.doc_ids,
            "use_vn_segmentation": self.use_vn_segmentation,
            "k1": config.BM25_K1,
            "b": config.BM25_B,
            "_engine": self._engine,
        }
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info(f"Sparse index saved to {path} (bm25s)")

    def load(self, path: str | Path):
        import bm25s as bm25s_lib
        path = Path(path)
        self.bm25 = bm25s_lib.BM25.load(str(path), load_corpus=False)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.doc_ids = meta.get("doc_ids", [])
            self.use_vn_segmentation = meta.get("use_vn_segmentation", True)
            self._engine = meta.get("_engine", "bm25s")
        # corpus loaded separately by caller
        self.corpus = []
        scores = self.bm25.scores
        n_docs = scores.get('num_docs', 0) if isinstance(scores, dict) else (scores.shape[0] if hasattr(scores, 'shape') else 0)
        if not self.doc_ids:
            self.doc_ids = list(range(n_docs))
        vocab_size = len(self.bm25.vocab_dict) if hasattr(self.bm25, 'vocab_dict') else 0
        logger.info(f"Sparse index loaded: {n_docs} docs, {vocab_size:,} terms")


def rrf_fusion(
    *ranked_lists: list[tuple[int, float]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 500,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}
    for i, ranked in enumerate(ranked_lists):
        weight = weights[i] if weights else 1.0
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + weight / (k + rank + 1)
    sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return [(idx, rrf_scores[idx]) for idx in sorted_idx]


async def build_indexes(
    docs: list[dict],
    embedder,
    dense_path: str | Path | None = None,
    sparse_path: str | Path | None = None,
    use_vn_segmentation: bool = True,
):
    dense_idx = DenseIndex()
    sparse_idx = SparseIndex(use_vn_segmentation=use_vn_segmentation)

    texts = [d["text"] for d in docs]
    ids = np.arange(len(docs), dtype=np.int64)

    if dense_path and Path(dense_path).exists():
        dense_idx.load(dense_path)
    else:
        logger.info("Encoding documents for dense index...")
        batch_size = 32
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            embs = await embedder.embed(batch)
            all_embs.extend(embs)
        dense_idx.add(np.array(all_embs), ids)
        if dense_path:
            dense_idx.save(dense_path)

    if sparse_path and Path(sparse_path).exists():
        sparse_idx.load(sparse_path)
    else:
        logger.info("Building BM25 index...")
        sparse_idx.build(texts, ids.tolist())
        if sparse_path:
            sparse_idx.save(sparse_path)

    return dense_idx, sparse_idx
