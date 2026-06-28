import os, sys, time, asyncio, traceback, json
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
TEST_SET_PATH = "D:\\AI_RAG_LEGAL\\test_set.json"

async def main():
    ram("start")
    # Load first 3 questions from test_set.json
    try:
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        queries = [item["question"] for item in test_data[:30]]
        qids = [item["id"] for item in test_data[:30]]
    except Exception as e:
        print(f"Error loading test set: {e}")
        queries = [
            "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
            "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
            "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
        ]
        qids = [1, 2, 3]

    class LazyDocsList:
        def __init__(self, jsonl_path):
            self.path = jsonl_path
            self.offsets = []
            t0 = time.time()
            print(f"Building offset map for {jsonl_path}...")
            with open(jsonl_path, 'rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line: break
                    self.offsets.append(offset)
            print(f"Mapped {len(self.offsets):,} docs in {time.time()-t0:.2f}s")
            self.file = open(self.path, 'rb')
        
        def __len__(self): return len(self.offsets)
        
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return [self[i] for i in range(*idx.indices(len(self.offsets)))]
            if idx < 0: idx += len(self.offsets)
            self.file.seek(self.offsets[idx])
            return json.loads(self.file.readline())

    from src.core.config import config
    corpus_path = os.path.join(config.DATA_DIR, "processed", config.CORPUS_FILE)
    t0 = time.time(); docs = LazyDocsList(corpus_path)
    ram("after corpus")

    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.reranker.cross_encoder import CrossEncoderReranker, LLMReranker, TwoStageReranker
    from src.llm.gemma4_client import Gemma4Client
    from src.pipeline.orchestrator import LegalRAGPipeline

    print("Loading models to GPU...")
    embedder = HarrierEmbedding()
    llm = Gemma4Client()
    ce = CrossEncoderReranker()
    
    print("Warming up models to pre-allocate CUDA memory...")
    try:
        # Warm up embedder
        import torch
        dummy_q = "Đây là câu hỏi test để khởi tạo bộ nhớ CUDA."
        await embedder.embed([dummy_q])
        
        # Warm up reranker (cross-encoder)
        from src.core.base import RetrievedChunk
        dummy_chunk = RetrievedChunk(chunk_id="test", doc_id=1, article_id="1", doc_title="Test", content="Nội dung test.")
        await ce.rerank(dummy_q, [dummy_chunk], top_k=1)
        
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warm up error: {e}")
        
    ram("after models")

    from src.retrieval.indexing import DenseIndex, SparseIndex
    di = DenseIndex(); di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    si = SparseIndex(); si.load(SPARSE)
    print(f"Indexes: dense={di.index.ntotal:,} sparse={len(si.doc_ids):,}")
    ram("after indexes")
    # reranker = TwoStageReranker(ce, LLMReranker(llm))
    reranker = ce  # Use only CrossEncoder for fast testing
    pipeline = LegalRAGPipeline(
        docs=docs, dense_index=di, sparse_index=si,
        embedder=embedder, llm=llm, reranker=reranker,
    )
    ram("pipeline ready")

    results = []
    for qid, q in zip(qids, queries):
        print("\n" + "=" * 100 + f"\nQUERY {qid}: {q}\n" + "=" * 100)
        try:
            t0 = time.time()
            r = await pipeline.answer_agentic(q)
            dt = time.time() - t0
            print(f"\n[type={r.query_type}] [{dt:.1f}s]")
            print(f"\n--- ANSWER ---\n{r.final_answer}\n")
            print(f"--- relevant_docs ({len(r.relevant_docs)}): {r.relevant_docs[:10]}")
            print(f"--- relevant_articles ({len(r.relevant_articles)}): {r.relevant_articles[:10]}")
            
            # Format to dict
            results.append({
                "id": qid,
                "question": q,
                "answer": r.final_answer,
                "relevant_docs": r.relevant_docs,
                "relevant_articles": r.relevant_articles
            })
        except Exception as e:
            print(f"\n!!! PIPELINE ERROR ON QUERY {qid}: {type(e).__name__}: {e}")
            traceback.print_exc()
            
    # Dump to output.json
    output_path = "D:\\AI_RAG_LEGAL\\scripts\\output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully wrote results to {output_path}")
    ram("done")

if __name__ == "__main__":
    asyncio.run(main())
