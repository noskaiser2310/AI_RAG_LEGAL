"""
VMTEB-ALQAC agent eval - độc lập theo batch.
Mỗi lần chạy xử lý 1 batch duy nhất, lưu file riêng.
--batch-idx N: xử lý batch thứ N (0-indexed, mỗi batch 100 queries)
--workers: số workers concurrent (mặc định 10)
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_batch")
QUERY_TIMEOUT = 600
KEY_COOLDOWN = 1.0  # seconds between requests per key


def load_vmteb_queries():
    from datasets import load_dataset
    logger.info("Loading VMTEB-ALQAC...")
    queries_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "queries", split="train")
    ir_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "data_ir", split="train")
    query_text = {q["query_id"]: q["question"] for q in queries_ds}
    qid_to_articles = defaultdict(set)
    for item in ir_ds:
        qid_to_articles[item["query_id"]].add(item["corpus_id"])
    eval_set = []
    for qid in sorted(query_text.keys()):
        eval_set.append({
            "query_id": qid,
            "question": query_text[qid],
            "reference_articles": sorted(qid_to_articles.get(qid, set())),
        })
    logger.info(f"Loaded {len(eval_set)} queries")
    return eval_set


def evaluate_retrieval(y_true, y_pred):
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
    results = {"macro_f2": macro_f2, "micro_f2": micro_f2, "micro_precision": micro_precision, "micro_recall": micro_recall, "mrr": mrr, "num_queries": n}
    for k in [1, 5, 10, 20, 50]:
        recalls = []
        for i in range(n):
            pred_k = set(y_pred_norm[i][:k])
            true_set = set(y_true_norm[i])
            if y_true[i]:
                recalls.append(len(pred_k & true_set) / len(y_true_norm[i]))
        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
    return results


async def process_one(item, pipeline, sem):
    async with sem:
        q = item["question"]
        t_start = time.time()
        try:
            result = await asyncio.wait_for(
                pipeline.answer(q, use_self_correct=False),
                timeout=QUERY_TIMEOUT,
            )
            elapsed = time.time() - t_start
            pred_ids = []
            seen = set()
            for c in result.chunks:
                so_ky_hieu = c.metadata.get("so_ky_hieu", "")
                article_id = c.article_id
                if so_ky_hieu and article_id:
                    corpus_id = f"{so_ky_hieu}#{article_id}"
                    if corpus_id not in seen:
                        seen.add(corpus_id)
                        pred_ids.append(corpus_id)
            logger.info(f"  OK {item['query_id']}: {len(pred_ids)} preds, {elapsed:.0f}s")
            trace = {
                "query_id": item["query_id"],
                "question": q,
                "reference_articles": item["reference_articles"],
                "predicted_articles": pred_ids[:50],
                "time": elapsed,
                "query_type": result.query_type,
                "confidence": result.confidence,
                "answer": result.final_answer or result.answer or "",
                "citations": result.citations,
                "relevant_articles": result.relevant_articles,
                "num_chunks": len(result.chunks),
            }
            return trace
        except asyncio.TimeoutError:
            logger.warning(f"  TIMEOUT {item['query_id']} ({QUERY_TIMEOUT}s)")
            return {"query_id": item["query_id"], "question": q, "reference_articles": item["reference_articles"], "predicted_articles": [], "error": "timeout", "time": QUERY_TIMEOUT}
        except Exception as e:
            msg = str(e)[:200]
            logger.error(f"  ERR {item['query_id']}: {msg}")
            return {"query_id": item["query_id"], "question": q, "reference_articles": item["reference_articles"], "predicted_articles": [], "error": msg, "time": time.time() - t_start}


async def main(args):
    from src.core.config import config
    from src.data.loading import load_corpus
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.retrieval.indexing import DenseIndex, SparseIndex
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.pipeline.agent import LegalAgent
    from src.retrieval.multi_strategy import MultiStrategyRetriever

    if args.llm == "hf":
        from src.llm.hf_client import HFClient
        llm = HFClient(model_name=args.hf_model, load_in_4bit=args.hf_4bit)
    else:
        from src.llm.gemini_parallel import GeminiLLM
        llm = GeminiLLM(max_concurrent=args.workers)

    docs = load_corpus(force_rebuild=False)
    logger.info(f"Corpus: {len(docs)} docs")
    embedder = HarrierEmbedding()
    ce = CrossEncoderReranker()
    logger.info("Loading indexes...")
    dense_idx = DenseIndex()
    dense_idx.load(str(Path(config.INDEX_DIR) / "dense.index"))
    sparse_idx = SparseIndex()
    sparse_idx.load(str(Path(config.INDEX_DIR) / "sparse"))
    retriever = MultiStrategyRetriever(docs, dense_idx, sparse_idx, embedder)
    pipeline = LegalAgent(llm=llm, retriever=retriever, reranker=ce, embedder=embedder, docs=docs)

    all_queries = load_vmteb_queries()
    batch_size = args.batch_size
    batch_idx = args.batch_idx
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(all_queries))
    batch = all_queries[batch_start:batch_end]

    logger.info(f"\nBatch {batch_idx}: queries {batch_start}-{batch_end-1} ({len(batch)} queries)")
    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.workers)
    predictions = []
    tasks = [process_one(item, pipeline, sem) for item in batch]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {batch_idx}"):
        predictions.append(await coro)

    await llm.close()

    # Evaluate
    valid = [p for p in predictions if "error" not in p]
    y_true = [p["reference_articles"] for p in valid]
    y_pred = [p["predicted_articles"] for p in valid]

    if y_true:
        results = evaluate_retrieval(y_true, y_pred)
    else:
        results = {"num_queries": 0, "error": "all queries failed"}
    results["batch_idx"] = batch_idx
    results["num_errors"] = len(predictions) - len(valid)
    results["avg_time"] = float(np.mean([p.get("time", 0) for p in predictions])) if predictions else 0.0

    metrics_path = output_dir / f"vmteb_batch{batch_idx}_metrics.json"
    detail_path = output_dir / f"vmteb_batch{batch_idx}_detail.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Save full trace per query (answer, citations, etc.)
    trace_path = output_dir / f"vmteb_batch{batch_idx}_trace.json"
    trace_data = [
        {k: p[k] for k in ("query_id", "question", "answer", "citations", "confidence", "query_type", "reference_articles", "predicted_articles", "error", "time")
         if k in p}
        for p in predictions
    ]
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH {batch_idx} DONE ({len(valid)}/{len(predictions)} OK)")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"Saved: {metrics_path}")
    print(f"       {detail_path}")
    print(f"       {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-idx", type=int, required=True, help="Batch index (0 = queries 0-99, 1 = 100-199, ...)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--llm", choices=["gemini", "hf"], default="gemini", help="LLM backend: gemini (API) or hf (local model)")
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-4B-Instruct", help="Model name khi --llm hf")
    parser.add_argument("--hf-4bit", action=argparse.BooleanOptionalAction, default=True, help="Load model local bang 4-bit")
    args = parser.parse_args()
    asyncio.run(main(args))
