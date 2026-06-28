# D:\AI_RAG_LEGAL\scripts\tune.py
"""Grid-search hyperparameters for LegalRAG pipeline on PBGDPL subset."""

import asyncio
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.core.config import config
from src.evaluation.metrics import compute_f2_macro, compute_retrieval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


import re
def _extract_article_ids(text: str) -> list[str]:
    return list(set(re.findall(r"Điều\s+(\d+)", text)))


def load_eval_set(max_queries: int = 30) -> tuple[list[str], list[list[str]]]:
    from datasets import load_dataset
    ds = load_dataset("tmquan/pbgdpl-vn-legal-qna", split="train", streaming=True)
    queries = []
    ground_truths = []
    for row in ds:
        q = row.get("question_text", row.get("question", ""))
        a = row.get("answer_text", row.get("answer", ""))
        if not q or not a:
            continue
        articles = list(set(re.findall(r"Điều\s+(\d+)", str(a))))
        if len(articles) >= 2:
            queries.append(str(q))
            ground_truths.append(articles)
        if len(queries) >= max_queries:
            break
    logger.info(f"Loaded {len(queries)} eval queries with ground truth")
    return queries, ground_truths


async def run_single_eval(
    pipeline, queries: list[str], ground_truths: list[list[str]], name: str
) -> dict:
    results = []
    for q, gt in zip(tqdm(queries, desc=name), ground_truths):
        try:
            result = await pipeline.answer(q)
            pred_articles = _extract_article_ids(result.final_answer)
            results.append({
                "query": q,
                "predicted_articles": pred_articles,
                "ground_truth": gt,
            })
        except Exception as e:
            logger.error(f"  Error: {q[:60]} -> {e}")
            results.append({"query": q, "predicted_articles": [], "ground_truth": gt})
    return results


def evaluate_trial(results: list[dict]) -> dict:
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted_articles"] for r in results]
    try:
        f2 = compute_f2_macro(y_true, y_pred)
        ret = compute_retrieval_metrics(y_true, y_pred, k_values=[5, 10])
    except Exception:
        return {"macro_f2": 0.0, "error": "eval failed"}
    return {
        "macro_f2": f2["macro_f2"],
        "micro_f2": f2["micro_f2"],
        "micro_recall": f2["micro_recall"],
        "micro_precision": f2["micro_precision"],
        "recall@5": ret.get("recall@5", 0),
        "recall@10": ret.get("recall@10", 0),
    }


async def main():
    from src.data.loading import load_corpus
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.llm.gemma4_client import Gemma4Client
    from src.pipeline.orchestrator import LegalRAGPipeline
    from src.reranker.cross_encoder import CrossEncoderReranker

    embedder = HarrierEmbedding()
    llm = Gemma4Client()
    reranker = CrossEncoderReranker()

    eval_queries, eval_ground_truths = load_eval_set(max_queries=30)

    logger.info("Loading corpus...")
    docs = load_corpus(force_rebuild=False)

    dense_weights = [float(w) for w in config.TUNE_DENSE_WEIGHTS.split(",")]
    score_thresholds = [float(s) for s in config.TUNE_SCORE_THRESHOLDS.split(",")]
    bm25_k1_values = [float(k) for k in config.TUNE_BM25_K1_VALUES.split(",")]
    gap_ratios = [float(g) for g in config.TUNE_GAP_RATIOS.split(",")]

    all_combos = list(itertools.product(dense_weights, score_thresholds, bm25_k1_values, gap_ratios))
    max_combos = min(len(all_combos), config.TUNE_MAX_COMBOS)
    all_combos = all_combos[:max_combos]

    logger.info(f"Tuning {len(all_combos)} parameter combinations on {len(eval_queries)} queries")

    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        embedder=embedder,
        llm=llm,
        reranker=reranker,
    )

    base_results = await run_single_eval(pipeline, eval_queries, eval_ground_truths, "Baseline")
    base_eval = evaluate_trial(base_results)
    logger.info(f"Baseline Macro-F2: {base_eval['macro_f2']:.4f}")

    all_trial_results = []
    for dw, st, bk1, gr in all_combos:
        try:
            from src.retrieval.multi_strategy import RRF_WEIGHTS_DENSE, adaptive_k_by_gap

            orig_st = config.SCORE_THRESHOLD
            orig_k1 = config.BM25_K1

            config.SCORE_THRESHOLD = st
            config.BM25_K1 = bk1

            trial_name = f"dw={dw:.2f}_st={st:.2f}_k1={bk1:.2f}_gr={gr:.2f}"

            trial_results = await run_single_eval(pipeline, eval_queries, eval_ground_truths, trial_name)
            trial_eval = evaluate_trial(trial_results)

            trial_entry = {
                "dense_weight": dw,
                "score_threshold": st,
                "bm25_k1": bk1,
                "gap_ratio": gr,
                **trial_eval,
            }
            all_trial_results.append(trial_entry)
            logger.info(f"  {trial_name}: F2={trial_eval['macro_f2']:.4f} R@5={trial_eval.get('recall@5',0):.4f}")

            config.SCORE_THRESHOLD = orig_st
            config.BM25_K1 = orig_k1
        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            continue

    if all_trial_results:
        best = max(all_trial_results, key=lambda x: x["macro_f2"])
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST: {json.dumps(best, indent=2)}")
        logger.info(f"{'='*50}")

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    tune_path = output_dir / "tuning_results.json"
    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": base_eval,
            "best": best if all_trial_results else None,
            "trials": all_trial_results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Tuning results saved to {tune_path}")

    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline F2: {base_eval['macro_f2']:.4f}")
    if best:
        print(f"Best F2:     {best['macro_f2']:.4f}")
        print(f"Best params: dense_weight={best['dense_weight']}, score_threshold={best['score_threshold']}, "
              f"bm25_k1={best['bm25_k1']}, gap_ratio={best['gap_ratio']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
