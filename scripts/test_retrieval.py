"""RAM-safe end-to-end RETRIEVAL test (dense mmap + BM25 + RRF + cross-encoder rerank).

Khong load full corpus vao RAM: dung byte-offset lookup tren corpus.jsonl.
Chay duoc tren may local (8GB GPU). Khong can LLM (chi test phan IR = phan tinh F2-macro).

    python scripts/test_retrieval.py
"""
import os, sys, json, time, asyncio
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import psutil, numpy as np, faiss
from src.core.base import RetrievedChunk
from src.core.config import config
from src.retrieval.indexing import DenseIndex, SparseIndex, rrf_fusion
from src.retrieval.text_processor import normalize_vietnamese_light, expand_legal_abbreviations

proc = psutil.Process()
def ram(tag): print(f"[RAM] {tag}: rss={proc.memory_info().rss/1e9:.2f}GB avail={psutil.virtual_memory().available/1e9:.2f}GB")

CORPUS = str(Path(config.DATA_DIR) / "processed" / config.CORPUS_FILE)
DENSE = str(Path(config.INDEX_DIR) / "dense.index")
SPARSE = str(Path(config.INDEX_DIR) / "sparse")
E5 = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"

QUERIES = [
    "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
    "Hộ kinh doanh cần đáp ứng điều kiện gì để được cấp giấy chứng nhận đăng ký?",
    "Người lao động có được hưởng lương trong thời gian thử việc không?",
]

def build_offsets(path):
    offsets = []
    with open(path, "rb") as f:
        off = f.tell(); line = f.readline()
        while line:
            offsets.append(off); off = f.tell(); line = f.readline()
    return offsets

def main():
    ram("start")
    t0 = time.time(); offsets = build_offsets(CORPUS)
    print(f"Corpus offsets: {len(offsets):,} dòng ({time.time()-t0:.1f}s)")

    def get_doc(i):
        with open(CORPUS, "rb") as f:
            f.seek(offsets[i]); return json.loads(f.readline())

    def to_chunk(i, score, source):
        d = get_doc(int(i))
        return RetrievedChunk(
            chunk_id=str(i), doc_id=d.get("doc_id", ""), article_id=str(d.get("article_id", "")),
            doc_title=d.get("title", ""), content=d.get("text", ""),
            score=float(score), retrieval_score=float(score), source=source,
            metadata={"so_ky_hieu": d.get("so_ky_hieu", ""), "source": d.get("source", "")},
        )

    # Dense (mmap) + Sparse (bm25s)
    t0 = time.time()
    di = DenseIndex(); di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    print(f"FAISS dense (mmap): {di.index.ntotal:,} vectors ({time.time()-t0:.1f}s)")
    t0 = time.time()
    si = SparseIndex(); si.load(SPARSE)
    print(f"BM25 sparse: {len(si.doc_ids):,} docs ({time.time()-t0:.1f}s)")
    ram("after indexes")

    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.reranker.cross_encoder import CrossEncoderReranker
    t0 = time.time(); emb = HarrierEmbedding(); print(f"Harrier loaded ({time.time()-t0:.1f}s)")
    t0 = time.time(); ce = CrossEncoderReranker(); print(f"CrossEncoder loaded ({time.time()-t0:.1f}s)")
    ram("after models")

    TOP_DENSE, TOP_SPARSE, TOP_RERANK = 50, 50, 5
    for qi, q in enumerate(QUERIES):
        print("\n" + "=" * 100)
        print(f"[Q{qi+1}] {q}")
        print("=" * 100)
        prep = expand_legal_abbreviations(normalize_vietnamese_light(q))

        t0 = time.time()
        qv = np.array(asyncio.run(emb.embed([f"{E5}\nTruy vấn: {prep}"])), dtype=np.float32)
        dense = di.search(qv, TOP_DENSE)
        sparse = si.search(prep, TOP_SPARSE)
        fused = rrf_fusion(dense, sparse, weights=[0.8, 0.2], top_k=20)
        t_ret = time.time() - t0

        print(f"\n--- RRF fusion (dense {len(dense)} + BM25 {len(sparse)}) — top 8 ứng viên [{t_ret:.1f}s] ---")
        cand = [to_chunk(i, s, "fused") for i, s in fused]
        for r, c in enumerate(cand[:8]):
            print(f"  {r+1}. [{c.metadata['so_ky_hieu'] or c.doc_title}] Điều {c.article_id} ({c.metadata['source']})")
            print(f"     {c.content[:130].replace(chr(10),' ')}")

        t0 = time.time()
        reranked = asyncio.run(ce.rerank(q, cand, TOP_RERANK))
        t_rr = time.time() - t0
        print(f"\n--- Cross-Encoder rerank — top {TOP_RERANK} [{t_rr:.1f}s]  => relevant_articles ---")
        for r, c in enumerate(reranked):
            print(f"  {r+1}. [score={c.rerank_score:.3f}] {c.metadata['so_ky_hieu'] or c.doc_title} | Điều {c.article_id}")
            print(f"     {c.content[:160].replace(chr(10),' ')}")

    ram("done")

if __name__ == "__main__":
    main()
