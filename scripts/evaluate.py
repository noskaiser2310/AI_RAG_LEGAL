import asyncio
import json
import logging
import re
import time
from pathlib import Path

from tqdm import tqdm

from src.core.config import config
from src.evaluation.metrics import compute_f2_macro, compute_retrieval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_article_ids(text: str) -> list[str]:
    return list(set(re.findall(r"Điều\s+(\d+)", text)))


async def main():
    from src.data.loading import load_pbgdpl_qa, load_corpus
    from src.pipeline.orchestrator import LegalRAGPipeline

    docs = load_corpus(force_rebuild=False)

    logger.info("Loading PBGDPL evaluation set...")
    eval_raw = load_pbgdpl_qa(max_qa=config.EVAL_MAX_QUERIES)

    eval_set = []
    for item in eval_raw:
        question = item.get("title", "")
        answer = item.get("text", "")
        if not question or not answer:
            continue
        ref_articles = extract_article_ids(answer)
        eval_set.append({
            "question": question,
            "reference_answer": answer,
            "reference_articles": ref_articles,
        })

    if not eval_set:
        logger.error("No evaluation data loaded!")
        return

    n_queries = min(len(eval_set), config.EVAL_MAX_QUERIES)
    eval_set = eval_set[:n_queries]
    logger.info(f"Evaluation set: {n_queries} queries")

    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        use_decomposition=True,
    )

    predictions = []
    for item in tqdm(eval_set, desc="Evaluating"):
        q = item["question"]
        t0 = time.time()
        try:
            result = await pipeline.answer(q)
            elapsed = time.time() - t0
            pred_articles = extract_article_ids(result.final_answer)
            predictions.append({
                "query": q,
                "reference_articles": item["reference_articles"],
                "predicted_articles": pred_articles,
                "confidence": result.confidence,
                "total_time": elapsed,
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

    if not y_true:
        logger.error("No valid predictions!")
        return

    f2 = compute_f2_macro(y_true, y_pred)
    ret_metrics = compute_retrieval_metrics(y_true, y_pred, k_values=[5, 10, 20, 50])

    results = {**f2, **ret_metrics}
    results["num_queries"] = len(y_true)
    results["num_errors"] = len(predictions) - len(y_true)

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "evaluation.json"
    detail_path = output_dir / "evaluation_detail.json"

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"Results: {eval_path}")

    print(f"\n{'='*50}")
    print(f"EVALUATION ({len(y_true)} queries)")
    print(f"{'='*50}")
    print(f"Macro-F2:    {results['macro_f2']:.4f}")
    print(f"Micro-F2:    {results['micro_f2']:.4f}")
    print(f"Micro-Recall:{results['micro_recall']:.4f}")
    for k in [5, 10, 20, 50]:
        print(f"Recall@{k}:  {results.get(f'recall@{k}', 0):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
