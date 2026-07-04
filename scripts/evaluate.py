"""
Evaluate on PBGDPL benchmark (Vietnamese Legal Q&A, 4,593 pairs).

Usage:
    python scripts/evaluate.py                          # default 200 queries
    python scripts/evaluate.py --max-queries 50
    python scripts/evaluate.py --mode retrieval         # retrieval only (no LLM)
    python scripts/evaluate.py --mode full              # full pipeline + LLM
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_article_ids(text: str) -> list[str]:
    return list(set(re.findall(r"Điều\s+(\d+)", str(text))))


def extract_articles_from_chunks(chunks) -> list[str]:
    articles = []
    seen = set()
    for c in chunks:
        if c.article_id and c.article_id not in seen:
            articles.append(c.article_id)
            seen.add(c.article_id)
    return articles


async def main(args):
    from src.data.loading import load_corpus, load_pbgdpl_qa
    from src.core.config import config
    from src.evaluation.metrics import compute_f2_macro, compute_retrieval_metrics, compute_mrr

    docs = load_corpus(force_rebuild=False)
    logger.info(f"Corpus: {len(docs)} docs")

    logger.info("Loading PBGDPL evaluation set...")
    eval_raw = load_pbgdpl_qa(max_qa=args.max_queries or None)

    eval_set = []
    for item in eval_raw:
        question = item.get("question", item.get("title", ""))
        answer = item.get("text", "")
        if not question or not answer:
            continue
        ref_articles = extract_article_ids(answer)
        if ref_articles:
            eval_set.append({
                "question": question,
                "reference_answer": answer,
                "reference_articles": ref_articles,
            })

    if args.max_queries > 0:
        eval_set = eval_set[:args.max_queries]

    logger.info(f"PBGDPL eval set: {len(eval_set)} queries (with article refs)")
    if not eval_set:
        logger.error("No evaluation data!")
        return

    if args.mode == "retrieval":
        # Retrieval-only: dense + BM25 + RRF + reranker, no LLM
        from src.embedding.harrier_embedding import HarrierEmbedding
        from src.retrieval.indexing import DenseIndex, SparseIndex

        embedder = HarrierEmbedding()
        di = DenseIndex(); di.load(str(Path(config.INDEX_DIR) / "dense.index"))
        si = SparseIndex(); si.load(str(Path(config.INDEX_DIR) / "sparse"))
        from src.reranker.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker()

        predictions = []
        for item in tqdm(eval_set, desc="PBGDPL (retrieval)"):
            q = item["question"]
            try:
                import numpy as np
                from src.retrieval.indexing import rrf_fusion
                from src.core.base import RetrievedChunk
                E5 = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"
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
                            source="hybrid", metadata={},
                        ))
                reranked = await ce.rerank(q, chunks, 50)
                pred_articles = extract_articles_from_chunks(reranked)
                predictions.append({
                    "query": q,
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": pred_articles,
                })
            except Exception as e:
                logger.error(f"Error: {q[:60]} -> {e}")
                predictions.append({
                    "query": q,
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": [],
                    "error": str(e),
                })
    else:
        # Full pipeline with LLM (Gemma4 via Gemini API)
        from src.llm.gemma4_client import Gemma4Client
        from src.pipeline.orchestrator import LegalRAGPipeline
        llm = Gemma4Client()
        pipeline = await LegalRAGPipeline.create(
            docs=docs,
            dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
            sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
            llm=llm,
            use_decomposition=False,
            use_hyde=False,
            use_query_expansion=False,
            use_adaptive=False,
        )

        predictions = []
        for item in tqdm(eval_set, desc="PBGDPL (full)"):
            q = item["question"]
            try:
                result = await pipeline.answer(q, use_self_correct=False)
                pred_articles = extract_article_ids(result.final_answer)
                predictions.append({
                    "query": q,
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": pred_articles,
                    "confidence": result.confidence,
                })
            except Exception as e:
                logger.error(f"Error: {q[:60]} -> {e}")
                predictions.append({
                    "query": q,
                    "reference_articles": item["reference_articles"],
                    "predicted_articles": [],
                    "error": str(e),
                })

    valid = [p for p in predictions if "error" not in p]
    y_true = [p["reference_articles"] for p in valid]
    y_pred = [p["predicted_articles"] for p in valid]

    f2 = compute_f2_macro(y_true, y_pred)
    ret = compute_retrieval_metrics(y_true, y_pred, k_values=[5, 10, 20, 50])
    mrr = compute_mrr(y_true, y_pred)
    results = {**f2, **ret, "mrr": mrr, "num_queries": len(y_true), "num_errors": len(predictions) - len(y_true)}

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pbgdpl_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    detail_path = output_dir / "pbgdpl_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"PBGDPL EVALUATION ({len(y_true)} queries) — mode={args.mode}")
    print(f"{'='*50}")
    print(f"Macro-F2:    {results['macro_f2']:.4f}")
    print(f"Micro-F2:    {results['micro_f2']:.4f}")
    print(f"Micro-Recall:{results['micro_recall']:.4f}")
    print(f"MRR:         {results['mrr']:.4f}")
    for k in [5, 10, 20]:
        print(f"Recall@{k}:  {results.get(f'recall@{k}', 0):.4f}")
    print(f"{'='*50}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on PBGDPL benchmark")
    parser.add_argument("--max-queries", type=int, default=200, help="Number of queries (0=all)")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval",
                        help="retrieval-only (no LLM) or full pipeline")
    args = parser.parse_args()
    asyncio.run(main(args))
