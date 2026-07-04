import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset

from src.core.config import config
from src.evaluation.metrics import compute_f2_macro, compute_retrieval_metrics
from src.pipeline.orchestrator import LegalRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_expected_articles(reference) -> list[str]:
    """Parse reference articles from VLSP format into our format."""
    # This might need adjustment depending on the exact schema of VLSP 2025
    # Assumes reference is a list of strings or list of dicts
    articles = []
    if isinstance(reference, list):
        for item in reference:
            if isinstance(item, dict):
                # E.g. {"law_id": "123", "article_id": "45"}
                doc_id = item.get("law_id", "") or item.get("doc_id", "")
                art_id = item.get("article_id", "")
                if doc_id and art_id:
                    articles.append(art_id) # Simplify for now, assuming article IDs match our chunk IDs
                elif art_id:
                    articles.append(art_id)
            elif isinstance(item, str):
                articles.append(item)
    elif isinstance(reference, str):
        articles.append(reference)
    return articles


def format_doc_id(chunk) -> str | None:
    import re
    meta = chunk.metadata or {}
    so_ky_hieu = meta.get("so_ky_hieu", "")
    title = chunk.doc_title or ""
    title = re.sub(r"\s*-\s*Điều\s+\d+.*$", "", title).strip()
    if so_ky_hieu and title:
        return f"{so_ky_hieu}|{title}"
    if title:
        return title
    return None


def extract_predicted_articles(result) -> list[str]:
    """Extract article references from pipeline output."""
    from src.retrieval.text_processor import extract_structured_references
    answer_refs = extract_structured_references(result.final_answer or "")
    answer_articles = set()
    for ref in answer_refs:
        aid = ref["article_id"]
        answer_articles.add(aid)

    # Fallback to retrieved chunks if answer didn't extract well
    if not answer_articles:
        for chunk in result.chunks[:5]:
            if chunk.article_id:
                answer_articles.add(chunk.article_id)
                
    return list(answer_articles)


async def main(args):
    # 1. Setup LLM Client based on environment (Laptop vs Kaggle)
    if args.client == "gemma":
        from src.llm.gemma4_client import Gemma4Client
        logger.info("Using Gemma4Client (Prototyping/API mode for Laptop)")
        llm_client = Gemma4Client()
    else:
        from src.llm.hf_client import HFClient
        logger.info("Using HFClient (Offline/vLLM mode for Kaggle/GPU)")
        llm_client = HFClient()

    # 2. Load dataset
    eval_set = []
    if args.test_file:
        logger.info(f"Loading local test file: {args.test_file}")
        with open(args.test_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            # Adapt this to local file schema
            for item in raw_data:
                eval_set.append({
                    "id": item.get("id", len(eval_set)),
                    "question": item.get("question", item.get("text", "")),
                    "reference_articles": extract_expected_articles(item.get("relevant_articles", item.get("reference_articles", [])))
                })
    else:
        logger.info(f"Loading from HuggingFace datasets: {args.hf_dataset}")
        try:
            if args.hf_config:
                ds = load_dataset(args.hf_dataset, args.hf_config, split="test")
            else:
                ds = load_dataset(args.hf_dataset, split="test")
                
            for item in ds:
                # Assuming schema has 'question' and 'relevant_articles'
                eval_set.append({
                    "id": item.get("id", len(eval_set)),
                    "question": item.get("question", ""),
                    "reference_articles": extract_expected_articles(item.get("relevant_articles", []))
                })
        except Exception as e:
            logger.error(f"Failed to load dataset from HF: {e}")
            logger.error("Please provide a local file via --test-file instead.")
            return

    if args.max_queries > 0:
        eval_set = eval_set[:args.max_queries]
    
    logger.info(f"Loaded {len(eval_set)} queries for evaluation.")

    # 3. Initialize Pipeline
    from src.data.loading import load_corpus
    docs = load_corpus(force_rebuild=False)
    
    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        llm=llm_client,
        use_decomposition=True,
    )

    # 4. Run Evaluation
    predictions = []
    y_true = []
    y_pred = []
    
    for item in tqdm(eval_set, desc="Evaluating VLSP 2025"):
        q = item["question"]
        t0 = time.time()
        try:
            result = await pipeline.answer(q)
            elapsed = time.time() - t0
            
            pred_articles = extract_predicted_articles(result)
            ref_articles = item["reference_articles"]
            
            y_true.append(ref_articles)
            y_pred.append(pred_articles)
            
            predictions.append({
                "id": item["id"],
                "query": q,
                "reference_articles": ref_articles,
                "predicted_articles": pred_articles,
                "confidence": result.confidence,
                "total_time": elapsed,
            })
        except Exception as e:
            logger.error(f"Error processing query '{q[:60]}': {e}")
            predictions.append({
                "id": item["id"],
                "query": q,
                "error": str(e)
            })

    # 5. Compute Metrics
    if not y_true:
        logger.error("No valid predictions to evaluate!")
        return

    f2_results = compute_f2_macro(y_true, y_pred)
    
    # 6. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_path = output_dir / "vlsp2025_metrics.json"
    detail_path = output_dir / "vlsp2025_predictions.json"

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(f2_results, f, ensure_ascii=False, indent=2)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    print(f"\n{'='*50}")
    print(f"VLSP 2025 EVALUATION ({len(y_true)} queries)")
    print(f"Model Client: {args.client.upper()}")
    print(f"{'='*50}")
    print(f"Macro-F2:    {f2_results.get('macro_f2', 0):.4f}")
    print(f"Micro-F2:    {f2_results.get('micro_f2', 0):.4f}")
    print(f"Micro-Recall:{f2_results.get('micro_recall', 0):.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on VLSP 2025 Benchmark")
    parser.add_argument("--client", choices=["gemma", "hf"], default="gemma", 
                        help="LLM Client to use (gemma for API/Laptop, hf for Offline/Kaggle)")
    parser.add_argument("--hf-dataset", type=str, default="VLSP2025-LegalSML/Public-Test",
                        help="HuggingFace dataset path")
    parser.add_argument("--hf-config", type=str, default="",
                        help="HuggingFace dataset config (e.g. multichoice_questions) nếu có")
    parser.add_argument("--test-file", type=str, default="",
                        help="Local JSON file containing test queries (overrides hf-dataset)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries (0 for all)")
    parser.add_argument("--output-dir", type=str, default="data/results",
                        help="Output directory for metrics and predictions")
    
    args = parser.parse_args()
    asyncio.run(main(args))
