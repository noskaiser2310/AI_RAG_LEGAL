import asyncio
import json
import logging
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import faiss

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.core.config import config
from src.embedding.harrier_embedding import HarrierEmbedding
from src.reranker.cross_encoder import CrossEncoderReranker
from src.llm.gemma4_client import Gemma4Client
from src.pipeline.orchestrator import LegalRAGPipeline
from src.retrieval.indexing import DenseIndex, SparseIndex

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
)

logger = logging.getLogger("test_30")

DENSE = str(Path(config.INDEX_DIR) / "dense.index")
SPARSE = str(Path(config.INDEX_DIR) / "sparse")

class LazyDocsList:
    def __init__(self, jsonl_path):
        self.path = jsonl_path
        self.offsets = []
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

async def main():
    logger.info("Initializing components...")
    
    config.DEVICE = "cpu"
    corpus_path = os.path.join(config.DATA_DIR, "processed", config.CORPUS_FILE)
    docs = LazyDocsList(corpus_path)
    
    embedder = HarrierEmbedding()
    llm = Gemma4Client()
    reranker = CrossEncoderReranker()
    
    di = DenseIndex()
    di.index = faiss.read_index(DENSE, faiss.IO_FLAG_MMAP)
    si = SparseIndex()
    si.load(SPARSE)
    
    pipeline = LegalRAGPipeline(
        docs=docs, dense_index=di, sparse_index=si,
        embedder=embedder, llm=llm, reranker=reranker,
    )
    
    logger.info("Pipeline initialized successfully!")

    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    # Take the first 30 queries
    queries_to_test = test_set[:30]
    
    results = []

    for i, item in enumerate(queries_to_test):
        q_id = item["id"]
        q_text = item["question"]
        
        print(f"\n{'='*100}")
        print(f"QUERY {i+1} (ID: {q_id}): {q_text}")
        print(f"{'='*100}\n")

        try:
            # Answer using agentic RAG
            result = await pipeline.answer_agentic(q_text)
            
            ans = result.final_answer
            docs = result.relevant_docs
            articles = result.relevant_articles
            
            print("--- FINAL ANSWER ---")
            print(ans)
            print(f"\n--- relevant_docs ({len(docs)}): {docs}")
            print(f"--- relevant_articles ({len(articles)}): {articles}\n")
            
            results.append({
                "id": q_id,
                "question": q_text,
                "answer": ans,
                "relevant_docs": docs,
                "relevant_articles": articles,
                "expected_answer": item.get("answer", ""),
                "expected_docs": item.get("relevant_docs", []),
                "expected_articles": item.get("relevant_articles", [])
            })
            
        except Exception as e:
            logger.error(f"Error processing query {q_id}: {e}")
            results.append({
                "id": q_id,
                "question": q_text,
                "error": str(e)
            })
            
    # Save results
    output_path = "results_30_queries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Finished processing 30 queries. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
