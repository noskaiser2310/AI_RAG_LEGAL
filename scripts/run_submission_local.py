"""Chạy N câu đầu test_set.json qua full pipeline (oneshot), xuất submission + in mẫu soát format.
Dùng mmap dense để tránh OOM trên máy local.
"""
import os, sys, json, time, asyncio, traceback, zipfile
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, "D:\\AI_RAG_LEGAL")
sys.path.insert(0, "D:\\AI_RAG_LEGAL\\scripts")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import psutil, faiss
proc = psutil.Process()
def ram(t): print(f"[RAM] {t}: rss={proc.memory_info().rss/1e9:.2f}GB avail={psutil.virtual_memory().available/1e9:.2f}GB", flush=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TEST = "D:\\AI_RAG_LEGAL\\test_set.json"
DENSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\dense.index"
SPARSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\sparse"
OUT = "D:\\AI_RAG_LEGAL\\data\\results"

async def main():
    from submit import build_submission_entry
    from src.data.loading import load_corpus
    from src.retrieval.indexing import DenseIndex, SparseIndex
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.reranker.cross_encoder import CrossEncoderReranker, LLMReranker, TwoStageReranker
    from src.llm.gemma4_client import Gemma4Client
    from src.pipeline.orchestrator import LegalRAGPipeline

    with open(TEST, encoding="utf-8") as f:
        test_data = json.load(f)[:N]
    print(f"Test queries: {len(test_data)}", flush=True)

    t0 = time.time(); docs = load_corpus(); print(f"Corpus: {len(docs):,} ({time.time()-t0:.0f}s)", flush=True)
    di = DenseIndex(); di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    si = SparseIndex(); si.load(SPARSE)
    embedder = HarrierEmbedding(); llm = Gemma4Client(); ce = CrossEncoderReranker()
    reranker = TwoStageReranker(ce, LLMReranker(llm))
    pipeline = LegalRAGPipeline(docs=docs, dense_index=di, sparse_index=si,
                               embedder=embedder, llm=llm, reranker=reranker)
    ram("ready")

    entries = []
    for item in test_data:
        qid, q = item["id"], item["question"]
        t0 = time.time()
        try:
            r = await pipeline.answer_agentic(q)
            e = build_submission_entry(qid, q, r)
            entries.append(e)
            print(f"[{qid}] OK type={r.query_type} docs={len(e['relevant_docs'])} "
                  f"articles={len(e['relevant_articles'])} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"[{qid}] ERROR: {type(ex).__name__}: {ex} ({time.time()-t0:.0f}s)", flush=True)
            traceback.print_exc()
            entries.append({"id": qid, "question": q, "answer": "",
                           "relevant_docs": [], "relevant_articles": []})

    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, "results.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(os.path.join(OUT, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rp, "results.json")
    print(f"\nSaved {rp} + submission.zip", flush=True)

    # In mẫu 2 entry để soát format
    print("\n===== SAMPLE ENTRIES (soát format) =====", flush=True)
    for e in entries[:2]:
        print(f"\n--- id={e['id']} ---", flush=True)
        print(f"question: {e['question']}", flush=True)
        print(f"answer[:200]: {(e['answer'] or '')[:200]}", flush=True)
        print(f"relevant_docs ({len(e['relevant_docs'])}):", flush=True)
        for d in e['relevant_docs'][:6]: print(f"    {d}", flush=True)
        print(f"relevant_articles ({len(e['relevant_articles'])}):", flush=True)
        for a in e['relevant_articles'][:8]: print(f"    {a}", flush=True)
    ram("done")

if __name__ == "__main__":
    asyncio.run(main())
