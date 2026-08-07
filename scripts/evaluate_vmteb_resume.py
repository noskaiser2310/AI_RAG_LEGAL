"""
VMTEB-ALQAC agent eval with resume & retry.
- --batch-idx N: xử lý batch thứ N
- Lần đầu: chạy full 50 queries
- Lần sau: chỉ chạy lại query bị lỗi (timeout, exception, hoặc empty predictions)
- Merge kết quả cũ + mới, lưu metrics cập nhật
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_resume")
QUERY_TIMEOUT = 600

RETRY_ERRORS = {"timeout", "ALL_FAILED"}


def load_vmteb_queries():
    from datasets import load_dataset
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
    return eval_set


def evaluate_retrieval(y_true, y_pred):
    def norm(x):
        return x.lower()
    yt_n = [[norm(a) for a in arts] for arts in y_true]
    yp_n = [[norm(a) for a in arts] for arts in y_pred]
    all_arts = sorted(set(norm(a) for arts in y_true for a in arts))
    a2i = {a: i for i, a in enumerate(all_arts)}
    n, na = len(y_true), len(all_arts)
    yt_b = np.zeros((n, na), dtype=int)
    yp_b = np.zeros((n, na), dtype=int)
    for i in range(n):
        for a in yt_n[i]:
            if a in a2i: yt_b[i, a2i[a]] = 1
        for a in yp_n[i]:
            if a in a2i: yp_b[i, a2i[a]] = 1
    r = {"num_queries": n}
    if n == 0:
        return r
    r["macro_f2"] = float(np.mean([fbeta_score(yt_b[i], yp_b[i], beta=2, zero_division=0) for i in range(n)]))
    r["micro_f2"] = float(fbeta_score(yt_b, yp_b, beta=2, average="micro", zero_division=0))
    r["micro_precision"] = float(precision_score(yt_b, yp_b, average="micro", zero_division=0))
    r["micro_recall"] = float(recall_score(yt_b, yp_b, average="micro", zero_division=0))
    rrs = []
    for i in range(n):
        ts = set(yt_n[i])
        for rank, a in enumerate(yp_n[i], 1):
            if a in ts:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    r["mrr"] = float(np.mean(rrs)) if rrs else 0.0
    for k in [1, 5, 10, 20, 50]:
        recs = []
        for i in range(n):
            pk = set(yp_n[i][:k])
            ts = set(yt_n[i])
            if y_true[i]:
                recs.append(len(pk & ts) / len(yt_n[i]))
        r[f"recall@{k}"] = float(np.mean(recs)) if recs else 0.0
    return r


async def process_one(item, pipeline, sem):
    async with sem:
        q = item["question"]
        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                pipeline.answer(q, use_self_correct=False), timeout=QUERY_TIMEOUT
            )
            el = time.time() - t0
            pred_ids = []
            seen = set()
            for c in result.chunks:
                skh = c.metadata.get("so_ky_hieu", "")
                aid = c.article_id
                if skh and aid:
                    cid = f"{skh}#{aid}"
                    if cid not in seen:
                        seen.add(cid)
                        pred_ids.append(cid)
            logger.info(f"  OK {item['query_id']}: {len(pred_ids)} preds, {el:.0f}s")
            return {"query_id": item["query_id"], "reference_articles": item["reference_articles"], "predicted_articles": pred_ids[:50], "time": el}
        except asyncio.TimeoutError:
            logger.warning(f"  TIMEOUT {item['query_id']} ({QUERY_TIMEOUT}s)")
            return {"query_id": item["query_id"], "reference_articles": item["reference_articles"], "predicted_articles": [], "error": "timeout", "time": QUERY_TIMEOUT}
        except Exception as e:
            msg = str(e)[:200]
            logger.error(f"  ERR {item['query_id']}: {msg}")
            return {"query_id": item["query_id"], "reference_articles": item["reference_articles"], "predicted_articles": [], "error": msg, "time": time.time() - t0}


async def main(args):
    from src.core.config import config
    from src.data.loading import load_corpus
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.retrieval.indexing import DenseIndex, SparseIndex
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.pipeline.agent import LegalAgent
    from src.retrieval.multi_strategy import MultiStrategyRetriever

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"vmteb_batch{args.batch_idx}_detail.json"
    metrics_path = output_dir / f"vmteb_batch{args.batch_idx}_metrics.json"

    # Load previous results if exist
    existing = {}
    if detail_path.exists():
        with open(detail_path, encoding="utf-8") as f:
            for p in json.load(f):
                existing[p["query_id"]] = p

    all_queries = load_vmteb_queries()
    batch_size = args.batch_size
    batch_start = args.batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(all_queries))
    batch = all_queries[batch_start:batch_end]

    # Identify queries to run
    to_run = []
    for item in batch:
        qid = item["query_id"]
        if qid in existing:
            err = existing[qid].get("error", "")
            preds = existing[qid].get("predicted_articles", [])
            should_retry = err in RETRY_ERRORS or (err == "" and not preds)
            if should_retry:
                to_run.append(item)
                logger.info(f"  {qid}: retry (error='{err}', preds={len(preds)})")
            else:
                logger.info(f"  {qid}: skip (OK, {len(preds)} preds)")
        else:
            to_run.append(item)
            logger.info(f"  {qid}: new")

    if not to_run:
        logger.info("All queries OK, nothing to run!")
        return

    logger.info(f"\nBatch {args.batch_idx}: {len(to_run)} queries to run ({batch_start}-{batch_end-1})")

    docs = load_corpus(force_rebuild=False)
    if args.llm == "hf":
        from src.llm.hf_client import HFClient
        llm = HFClient(model_name=args.hf_model, load_in_4bit=args.hf_4bit)
    else:
        from src.llm.gemini_parallel import GeminiLLM
        llm = GeminiLLM(max_concurrent=args.workers)
    embedder = HarrierEmbedding()
    ce = CrossEncoderReranker()
    dense_idx = DenseIndex()
    dense_idx.load(str(Path(config.INDEX_DIR) / "dense.index"))
    sparse_idx = SparseIndex()
    sparse_idx.load(str(Path(config.INDEX_DIR) / "sparse"))
    from src.retrieval.multi_strategy import MultiStrategyRetriever
    retriever = MultiStrategyRetriever(docs, dense_idx, sparse_idx, embedder)
    pipeline = LegalAgent(llm=llm, retriever=retriever, reranker=ce, embedder=embedder, docs=docs)

    sem = asyncio.Semaphore(args.workers)
    new_results = []
    tasks = [process_one(item, pipeline, sem) for item in to_run]
    from tqdm import tqdm
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {args.batch_idx}"):
        new_results.append(await coro)

    await llm.close()

    # Merge: old OK results + new results
    merged = []
    seen_qids = set()
    for item in batch:
        qid = item["query_id"]
        if qid in existing and qid not in {r["query_id"] for r in new_results}:
            merged.append(existing[qid])
        seen_qids.add(qid)
    for r in new_results:
        merged.append(r)

    # Save detail
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Eval
    valid = [p for p in merged if "error" not in p]
    if valid:
        y_true = [p["reference_articles"] for p in valid]
        y_pred = [p["predicted_articles"] for p in valid]
        results = evaluate_retrieval(y_true, y_pred)
    else:
        results = {"num_queries": 0}
    results["batch_idx"] = args.batch_idx
    results["num_errors"] = len(merged) - len(valid)
    results["avg_time"] = float(np.mean([p.get("time", 0) for p in merged])) if merged else 0.0
    results["num_retried"] = len(to_run)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH {args.batch_idx} ({len(valid)}/{len(merged)} OK, retried={len(to_run)})")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-idx", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm", choices=["gemini", "hf"], default="gemini", help="LLM backend: gemini (API) or hf (local model)")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name khi --llm hf")
    parser.add_argument("--hf-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    asyncio.run(main(args))

