import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vmteb_queries(max_queries: int = 0):
    from datasets import load_dataset

    logger.info("Loading VMTEB-ALQAC queries...")
    queries_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "queries")["train"]
    ir_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "data_ir")["train"]

    query_text = {q["query_id"]: q["question"] for q in queries_ds}

    qid_to_articles = defaultdict(set)
    for item in ir_ds:
        qid_to_articles[item["query_id"]].add(item["corpus_id"])

    eval_set = []
    for qid in list(query_text.keys()):
        eval_set.append({
            "query_id": qid,
            "question": query_text[qid],
            "reference_articles": sorted(qid_to_articles.get(qid, set())),
        })

    if max_queries > 0:
        eval_set = eval_set[:max_queries]

    logger.info(f"Loaded {len(eval_set)} queries from VMTEB-ALQAC")
    return eval_set


def evaluate_retrieval(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    # Case-insensitive matching (VMTEB uses lowercase, our corpus uses uppercase)
    def normalize(x):
        return x.lower()

    y_true_norm = [[normalize(a) for a in articles] for articles in y_true]
    y_pred_norm = [[normalize(a) for a in articles] for articles in y_pred]

    all_articles = sorted(set(normalize(a) for articles in y_true for a in articles))
    article_to_idx = {a: i for i, a in enumerate(all_articles)}
    n = len(y_true)
    n_articles = len(all_articles)

    y_true_bin = np.zeros((n, n_articles), dtype=int)
    y_pred_bin = np.zeros((n, n_articles), dtype=int)

    for i in range(n):
        for a in y_true_norm[i]:
            if a in article_to_idx:
                y_true_bin[i, article_to_idx[a]] = 1
        for a in y_pred_norm[i]:
            if a in article_to_idx:
                y_pred_bin[i, article_to_idx[a]] = 1

    macro_f2 = float(np.mean([fbeta_score(y_true_bin[i], y_pred_bin[i], beta=2, zero_division=0) for i in range(n)]))
    micro_f2 = float(fbeta_score(y_true_bin, y_pred_bin, beta=2, average="micro", zero_division=0))
    micro_precision = float(precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0))

    mrr = 0.0
    rrs = []
    for i in range(n):
        true_set = set(y_true_norm[i])
        for rank, a in enumerate(y_pred_norm[i], 1):
            if a in true_set:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    mrr = float(np.mean(rrs)) if rrs else 0.0

    results = {
        "macro_f2": macro_f2,
        "micro_f2": micro_f2,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "mrr": mrr,
        "num_queries": n,
    }

    for k in [1, 5, 10, 20, 50]:
        recalls = []
        for i in range(n):
            pred_k = set(y_pred_norm[i][:k])
            true_set = set(y_true_norm[i])
            hits = len(pred_k & true_set)
            if y_true[i]:
                recalls.append(hits / len(y_true_norm[i]))
        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0

    return results


async def main(args):
    from src.core.config import config
    from src.data.loading import load_corpus

    docs = load_corpus(force_rebuild=False)
    logger.info(f"Corpus: {len(docs)} docs")

    eval_set = load_vmteb_queries(args.max_queries)
    if not eval_set:
        logger.error("No eval data")
        return

    if args.mode == "retrieval":
        from src.embedding.harrier_embedding import HarrierEmbedding
        from src.retrieval.indexing import DenseIndex, SparseIndex, rrf_fusion
        from src.reranker.cross_encoder import CrossEncoderReranker
        from src.core.base import RetrievedChunk

        embedder = HarrierEmbedding()
        di = DenseIndex(); di.load(str(Path(config.INDEX_DIR) / "dense.index"))
        si = SparseIndex(); si.load(str(Path(config.INDEX_DIR) / "sparse"))
        ce = CrossEncoderReranker()

        E5 = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"

        predictions = []
        for item in tqdm(eval_set, desc="VMTEB retrieval"):
            q = item["question"]
            try:
                import numpy as np
                embs = await embedder.embed([f"{E5}\nTruy vấn: {q}"])
                qv = np.array(embs[0], dtype=np.float32)
                dense = di.search(qv, 500)
                sparse = si.search(q, 500)
                fused = rrf_fusion(dense, sparse, weights=[0.8, 0.2], top_k=100)
                chunks = []
                for doc_id, score in fused:
                    if doc_id < len(docs):
                        d = docs[doc_id]
                        chunks.append(RetrievedChunk(
                            chunk_id=str(doc_id), doc_id=d.get("doc_id", ""),
                            article_id=str(d.get("article_id", "")),
                            doc_title=d.get("title", ""), content=d.get("text", ""),
                            score=float(score), retrieval_score=float(score), rerank_score=float(score),
                            source="hybrid",
                            metadata={"so_ky_hieu": d.get("so_ky_hieu", "")},
                        ))
                reranked = await ce.rerank(q, chunks, 50)

                pred_ids = []
                seen = set()
                for c in reranked:
                    corpus_id_key = f"{c.metadata.get('so_ky_hieu', '')}#{c.article_id}"
                    if c.article_id and corpus_id_key not in seen:
                        seen.add(corpus_id_key)
                        so_ky_hieu = c.metadata.get("so_ky_hieu", "")
                        if so_ky_hieu:
                            corpus_id = f"{so_ky_hieu}#{c.article_id}"
                        else:
                            corpus_id = c.article_id
                        pred_ids.append(corpus_id)

                predictions.append({
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": pred_ids,
                })
            except Exception as e:
                logger.error(f"Error: {str(e)[:100]}")
                predictions.append({
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": [],
                    "error": str(e),
                })
    else:
        from src.embedding.harrier_embedding import HarrierEmbedding
        from src.retrieval.indexing import DenseIndex, SparseIndex, rrf_fusion
        from src.reranker.cross_encoder import CrossEncoderReranker
        from src.core.base import RetrievedChunk
        from src.llm.gemini_parallel import GeminiParallelClient

        embedder = HarrierEmbedding()
        di = DenseIndex(); di.load(str(Path(config.INDEX_DIR) / "dense.index"))
        si = SparseIndex(); si.load(str(Path(config.INDEX_DIR) / "sparse"))
        ce = CrossEncoderReranker()
        llm = GeminiParallelClient(model=args.llm_model, max_concurrent=args.workers)

        E5 = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"
        SYSTEM_PROMPT = "Bạn là trợ lý pháp lý Việt Nam. Dựa vào các điều luật được cung cấp, hãy trả lời CHÍNH XÁC câu hỏi. Chỉ đưa ra câu trả lời ngắn gọn."

        async def process_one(item):
            q = item["question"]
            try:
                import numpy as np
                embs = await embedder.embed([f"{E5}\nTruy vấn: {q}"])
                qv = np.array(embs[0], dtype=np.float32)
                dense = di.search(qv, 500)
                sparse = si.search(q, 500)
                fused = rrf_fusion(dense, sparse, weights=[0.8, 0.2], top_k=100)
                chunks = []
                for doc_id, score in fused:
                    if doc_id < len(docs):
                        d = docs[doc_id]
                        chunks.append(RetrievedChunk(
                            chunk_id=str(doc_id), doc_id=d.get("doc_id", ""),
                            article_id=str(d.get("article_id", "")),
                            doc_title=d.get("title", ""), content=d.get("text", ""),
                            score=float(score), retrieval_score=float(score), rerank_score=float(score),
                            source="hybrid",
                            metadata={"so_ky_hieu": d.get("so_ky_hieu", "")},
                        ))
                reranked = await ce.rerank(q, chunks, 20)

                ctx_parts = []
                pred_ids = []
                seen = set()
                for c in reranked:
                    corpus_id_key = f"{c.metadata.get('so_ky_hieu', '')}#{c.article_id}"
                    if c.article_id and corpus_id_key not in seen:
                        seen.add(corpus_id_key)
                        so_ky_hieu = c.metadata.get("so_ky_hieu", "")
                        corpus_id = f"{so_ky_hieu}#{c.article_id}" if so_ky_hieu else c.article_id
                        pred_ids.append(corpus_id)
                        ctx_parts.append(f"Điều {c.article_id}:\n{c.content[:2000]}")

                context = "\n\n".join(ctx_parts[:10])
                prompt = f"Văn bản pháp luật:\n{context}\n\nCâu hỏi: {q}\n\nTrả lời:"
                answer, used_key = await llm.generate(prompt, SYSTEM_PROMPT)

                return {
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": pred_ids,
                    "answer": answer,
                    "used_key": used_key[:16] if used_key else "",
                }
            except Exception as e:
                return {
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": [],
                    "error": str(e),
                    "answer": "",
                }

        sem = asyncio.Semaphore(args.workers)
        async def worker(item):
            async with sem:
                return await process_one(item)

        predictions = []
        tasks = [worker(item) for item in eval_set]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="VMTEB full"):
            predictions.append(await coro)

        await llm.close()

    valid = [p for p in predictions if "error" not in p]
    y_true = [p["reference_articles"] for p in valid]
    y_pred = [p["predicted_articles"] for p in valid]

    results = evaluate_retrieval(y_true, y_pred)
    results["num_errors"] = len(predictions) - len(valid)

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vmteb_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    detail_path = output_dir / "vmteb_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"VMTEB EVALUATION ({len(y_true)} queries) — mode={args.mode}")
    print(f"{'='*50}")
    print(f"Macro-F2:     {results['macro_f2']:.4f}")
    print(f"Micro-F2:     {results['micro_f2']:.4f}")
    print(f"Micro-Prec:   {results['micro_precision']:.4f}")
    print(f"Micro-Recall: {results['micro_recall']:.4f}")
    print(f"MRR:          {results['mrr']:.4f}")
    for k in [1, 5, 10, 20]:
        print(f"Recall@{k}:   {results.get(f'recall@{k}', 0):.4f}")
    print(f"Errors:       {results.get('num_errors', 0)}")
    print(f"{'='*50}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on VMTEB-ALQAC benchmark")
    parser.add_argument("--max-queries", type=int, default=0, help="Number of queries (0=all 620)")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--llm-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    args = parser.parse_args()
    asyncio.run(main(args))
