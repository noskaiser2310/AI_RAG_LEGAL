"""
Background runner for VMTEB agent eval.
Saves progress incrementally so it can be resumed.
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_eval")

CHECKPOINT = Path("data/results/vmteb_agent_checkpoint.json")
METRICS_OUT = Path("data/results/vmteb_agent_metrics.json")
DETAIL_OUT = Path("data/results/vmteb_agent_detail.json")


async def main():
    from src.core.config import config
    from src.data.loading import load_corpus
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.pipeline.orchestrator import LegalRAGPipeline
    from src.llm.gemini_parallel import GeminiLLM

    MAX_WORKERS = 10

    docs = load_corpus(force_rebuild=False)
    logger.info(f"Corpus: {len(docs)} docs")

    # Load checkpoint if exists
    completed_ids = set()
    predictions = []
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            cp = json.load(f)
            completed_ids = set(cp.get("completed_ids", []))
            predictions = cp.get("predictions", [])
        logger.info(f"Resumed: {len(completed_ids)} already done")

    llm = GeminiLLM(max_concurrent=MAX_WORKERS)
    embedder = HarrierEmbedding()
    dense_path = str(Path(config.INDEX_DIR) / "dense.index")
    sparse_path = str(Path(config.INDEX_DIR) / "sparse")

    pipeline = await LegalRAGPipeline.create(
        docs=docs, dense_path=dense_path, sparse_path=sparse_path,
        embedder=embedder, llm=llm,
        use_hyde=True, use_query_expansion=True, use_adaptive=True, use_decomposition=True,
    )

    from datasets import load_dataset
    from collections import defaultdict

    logger.info("Loading VMTEB queries...")
    queries_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "queries", split="train")
    ir_ds = load_dataset("another-symato/VMTEB-ALQAC-retrieval", "data_ir", split="train")

    query_text = {q["query_id"]: q["question"] for q in queries_ds}
    qid_to_articles = defaultdict(set)
    for item in ir_ds:
        qid_to_articles[item["query_id"]].add(item["corpus_id"])

    eval_items = []
    for qid in list(query_text.keys()):
        if qid in completed_ids:
            continue
        eval_items.append({
            "query_id": qid,
            "question": query_text[qid],
            "reference_articles": sorted(qid_to_articles.get(qid, set())),
        })

    logger.info(f"Remaining: {len(eval_items)} queries")

    if not eval_items:
        logger.info("All done!")
        return

    sem = asyncio.Semaphore(MAX_WORKERS)

    async def process_one(item):
        async with sem:
            q = item["question"]
            t_start = time.time()
            try:
                result = await pipeline.answer_agentic(q, use_self_correct=False)
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
                return {
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": pred_ids[:50],
                    "answer": result.final_answer,
                    "time": elapsed,
                }
            except Exception as e:
                logger.error(f"  ERR {item['query_id']}: {str(e)[:200]}")
                return {
                    "query_id": item["query_id"],
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": [],
                    "error": str(e),
                    "answer": "",
                    "time": time.time() - t_start,
                }

    n_total = len(eval_items)
    for idx in range(0, n_total, MAX_WORKERS):
        batch = eval_items[idx : idx + MAX_WORKERS]
        tasks = [process_one(item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        predictions.extend(batch_results)

        # Save checkpoint every batch
        completed_ids.update(p["query_id"] for p in batch_results if "error" not in p)
        cp_data = {"completed_ids": list(completed_ids), "predictions": predictions}
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(cp_data, f, ensure_ascii=False)
        logger.info(f"Progress: {len(completed_ids)}/{n_total + len(completed_ids)}")

    await llm.close()
    _save_results(predictions)


def _save_results(predictions):
    import numpy as np
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    def normalize(x):
        return x.lower()

    valid = [p for p in predictions if "error" not in p]
    y_true = [p["reference_articles"] for p in valid]
    y_pred = [p["predicted_articles"] for p in valid]

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

    mrr = float(np.mean([next((1.0 / rank for rank, a in enumerate(y_pred_norm[i], 1) if a in set(y_true_norm[i])), 0.0) for i in range(n)]))

    results = {
        "macro_f2": macro_f2,
        "micro_f2": micro_f2,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "mrr": mrr,
        "num_queries": n,
        "num_errors": len(predictions) - len(valid),
        "avg_time": float(np.mean([p.get("time", 0) for p in predictions])),
    }
    for k in [1, 5, 10, 20, 50]:
        recalls = [len(set(y_pred_norm[i][:k]) & set(y_true_norm[i])) / max(len(y_true_norm[i]), 1) for i in range(n)]
        results[f"recall@{k}"] = float(np.mean(recalls))

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(DETAIL_OUT, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"VMTEB AGENT EVAL ({n} queries)")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {METRICS_OUT}")
    print(f"       {DETAIL_OUT}")

    CHECKPOINT.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
