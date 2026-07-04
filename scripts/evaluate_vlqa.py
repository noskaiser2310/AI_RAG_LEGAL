"""
Evaluate on VLQA benchmark (Vietnamese Legal Question Answering).

VLQA: 3,129 questions, 59,636 articles
Paper: arXiv:2507.19995 (Tan-Minh Nguyen et al., 2025)

Usage:
    python scripts/evaluate_vlqa.py                          # auto-detect HF dataset
    python scripts/evaluate_vlqa.py --local path/to/vlqa.json # local file
    python scripts/evaluate_vlqa.py --max-queries 100          # limit
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

from src.core.config import config
from src.evaluation.metrics import compute_f2_macro, compute_retrieval_metrics, compute_mrr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VLQA_HF_CANDIDATES = [
    "vlqa/vlqa",
    "JAIST/vlqa",
    "minhnt/vlqa",
    "vitnamese-legal/vlqa",
    "tmquan/vlqa",
    "UET/vlqa",
]


def extract_article_ids(text: str) -> list[str]:
    return list(set(re.findall(r"Điều\s+(\d+)", str(text))))


async def main(args):
    eval_set = []

    if args.local:
        logger.info(f"Loading local file: {args.local}")
        with open(args.local, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            q = item.get("question", item.get("question_text", ""))
            ref = item.get("relevant_articles", item.get("reference_articles", []))
            if isinstance(ref, str):
                ref = [ref]
            if q:
                eval_set.append({"question": q, "reference_articles": ref})
    else:
        from datasets import load_dataset

        ds = None
        for name in VLQA_HF_CANDIDATES:
            try:
                logger.info(f"Trying HF dataset: {name}...")
                ds = load_dataset(name, split="test", streaming=True)
                logger.info(f"Loaded: {name}")
                break
            except Exception:
                continue

        if ds is None:
            logger.error("VLQA not found on HuggingFace. Provide --local path.")
            logger.error("Check: https://huggingface.co/datasets?search=vlqa")
            return

        for row in ds:
            q = row.get("question", row.get("question_text", ""))
            ref = row.get("relevant_articles", row.get("reference_articles", []))
            if q:
                eval_set.append({"question": q, "reference_articles": ref})

    if args.max_queries > 0:
        eval_set = eval_set[:args.max_queries]

    logger.info(f"VLQA eval set: {len(eval_set)} queries")
    if not eval_set:
        logger.error("No queries loaded!")
        return

    from src.data.loading import load_corpus
    from src.pipeline.orchestrator import LegalRAGPipeline

    docs = load_corpus(force_rebuild=False)
    logger.info(f"Corpus: {len(docs)} docs")

    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        use_decomposition=True,
    )

    predictions = []
    for item in tqdm(eval_set, desc="VLQA"):
        q = item["question"]
        t0 = time.time()
        try:
            result = await pipeline.answer(q)
            pred_articles = extract_article_ids(result.final_answer)
            predictions.append({
                "query": q,
                "reference_articles": item["reference_articles"],
                "predicted_articles": pred_articles,
                "confidence": result.confidence,
                "total_time": time.time() - t0,
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
    path = output_dir / "vlqa_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(output_dir / "vlqa_detail.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"VLQA EVALUATION ({len(y_true)} queries)")
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
    parser = argparse.ArgumentParser(description="Evaluate on VLQA benchmark")
    parser.add_argument("--local", type=str, help="Path to local VLQA JSON file")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit queries (0=all)")
    args = parser.parse_args()
    asyncio.run(main(args))
