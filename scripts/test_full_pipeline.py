"""Smoke-test FULL pipeline (orchestrator): classify -> expand -> retrieve -> rerank -> generate.
Dùng mmap cho dense index để đỡ RAM. LLM = client cấu hình trong .env (Gemini/Gemma - prototyping).
"""
import os, sys, time, asyncio, traceback
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, "D:\\AI_RAG_LEGAL")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import psutil, faiss
proc = psutil.Process()
def ram(tag): print(f"[RAM] {tag}: rss={proc.memory_info().rss/1e9:.2f}GB avail={psutil.virtual_memory().available/1e9:.2f}GB")

DENSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\dense.index"
SPARSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\sparse"

async def main():
    ram("start")
    from src.data.loading import load_corpus
    t0 = time.time(); docs = load_corpus(); print(f"Corpus: {len(docs):,} docs ({time.time()-t0:.1f}s)")
    ram("after corpus")

    from src.retrieval.indexing import DenseIndex, SparseIndex
    di = DenseIndex(); di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    si = SparseIndex(); si.load(SPARSE)
    print(f"Indexes: dense={di.index.ntotal:,} sparse={len(si.doc_ids):,}")
    ram("after indexes")

    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.reranker.cross_encoder import CrossEncoderReranker, LLMReranker, TwoStageReranker
    from src.llm.gemma4_client import Gemma4Client
    from src.pipeline.orchestrator import LegalRAGPipeline

    embedder = HarrierEmbedding()
    llm = Gemma4Client()
    ce = CrossEncoderReranker()
    reranker = TwoStageReranker(ce, LLMReranker(llm))
    pipeline = LegalRAGPipeline(
        docs=docs, dense_index=di, sparse_index=si,
        embedder=embedder, llm=llm, reranker=reranker,
    )
    ram("pipeline ready")

    queries = ["Người lao động có được hưởng lương trong thời gian thử việc không?"]
    for q in queries:
        print("\n" + "=" * 100 + f"\nQUERY: {q}\n" + "=" * 100)
        try:
            t0 = time.time()
            r = await pipeline.answer(q)
            dt = time.time() - t0
            print(f"\n[type={r.query_type}] [{dt:.1f}s]")
            print(f"\n--- ANSWER ---\n{r.final_answer}\n")
            print(f"--- relevant_articles ({len(r.relevant_articles)}): {r.relevant_articles[:10]}")
            print(f"--- citations: {r.citations[:10]}")
            print(f"--- confidence: {r.confidence:.3f}  corrections: {r.num_correction_rounds}")
        except Exception as e:
            print(f"\n!!! PIPELINE ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
    ram("done")

if __name__ == "__main__":
    asyncio.run(main())
