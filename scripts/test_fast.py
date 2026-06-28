import os, sys, time, asyncio, traceback, json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, "D:\\AI_RAG_LEGAL")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DENSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\dense.index"
SPARSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\sparse"
TEST_SET_PATH = "D:\\AI_RAG_LEGAL\\test_set.json"

async def main():
    try:
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        queries = [item["question"] for item in test_data[:5]]
        qids = [item["id"] for item in test_data[:5]]
    except Exception as e:
        print(f"Error loading test set: {e}")
        return

    class LazyDocsList:
        def __init__(self, jsonl_path):
            self.path = jsonl_path
            self.offsets = []
            t0 = time.time()
            with open(jsonl_path, 'rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line: break
                    self.offsets.append(offset)
            self.file = open(self.path, 'rb')
        
        def __len__(self): return len(self.offsets)
        
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return [self[i] for i in range(*idx.indices(len(self.offsets)))]
            if idx < 0: idx += len(self.offsets)
            self.file.seek(self.offsets[idx])
            return json.loads(self.file.readline())

    from src.core.config import config
    logging.info("Initializing components...")
    
    config.DEVICE = "cpu"
    corpus_path = os.path.join(config.DATA_DIR, "processed", config.CORPUS_FILE)
    docs = LazyDocsList(corpus_path)

    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.llm.gemma4_client import Gemma4Client
    from src.pipeline.orchestrator import LegalRAGPipeline

    embedder = HarrierEmbedding()
    llm = Gemma4Client()
    ce = CrossEncoderReranker()

    from src.retrieval.indexing import DenseIndex, SparseIndex
    import faiss
    di = DenseIndex(); di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    si = SparseIndex(); si.load(SPARSE)

    pipeline = LegalRAGPipeline(
        docs=docs, dense_index=di, sparse_index=si,
        embedder=embedder, llm=llm, reranker=ce,
    )

    results = []
    for qid, q in zip(qids, queries):
        print(f"\n==================== QUERY {qid} ====================")
        try:
            t0 = time.time()
            r = await pipeline.answer_agentic(q)
            dt = time.time() - t0
            print(f"\n[TIME] {dt:.1f}s")
            print(f"--- ANSWER ---\n{r.final_answer}\n")
            print(f"--- relevant_docs ({len(r.relevant_docs)}): {r.relevant_docs}")
            print(f"--- relevant_articles ({len(r.relevant_articles)}): {r.relevant_articles}")
            
            results.append({
                "id": qid,
                "question": q,
                "answer": r.final_answer,
                "relevant_docs": r.relevant_docs,
                "relevant_articles": r.relevant_articles
            })
        except Exception as e:
            print(f"Error on query {qid}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
