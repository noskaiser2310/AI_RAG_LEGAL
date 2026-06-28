"""Export pipeline results to R2AI2026 competition submission format.

Format:
    {
        "id": <int>,
        "question": "<string>",
        "answer": "<string>",
        "relevant_docs": ["<mã VB>|<tên VB>"],
        "relevant_articles": ["<mã VB>|<tên VB>|<điều>"]
    }
"""
import asyncio
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path

from tqdm import tqdm

os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.core.config import config
from src.retrieval.text_processor import extract_doc_code, extract_structured_references

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_doc_id(chunk) -> str | None:
    meta = chunk.metadata or {}
    so_ky_hieu = meta.get("so_ky_hieu", "")
    title = chunk.doc_title or ""
    title = re.sub(r"\s*-\s*Điều\s+\d+.*$", "", title).strip()
    if so_ky_hieu and title:
        return f"{so_ky_hieu}|{title}"
    if title:
        return title
    return None


def format_article_ref(doc_str: str, article_id: str, clause_id: str = "") -> str:
    if clause_id:
        return f"{doc_str}|Điều {article_id} Khoản {clause_id}"
    return f"{doc_str}|Điều {article_id}"


def build_submission_entry(
    qid: int,
    question: str,
    result,
) -> dict:
    doc_refs = set()
    article_refs = set()

    chunk_by_article = {}
    for chunk in result.chunks:
        doc_str = format_doc_id(chunk)
        if doc_str:
            doc_refs.add(doc_str)
        if chunk.article_id and doc_str:
            chunk_by_article[chunk.article_id] = doc_str

    if result.relevant_docs:
        for d in result.relevant_docs:
            if d:
                code = extract_doc_code(d)
                if code:
                    doc_refs.add(f"{code}|{d}")
                else:
                    doc_refs.add(d)

    answer_refs = extract_structured_references(result.final_answer or "")
    answer_articles = set()
    for ref in answer_refs:
        aid = ref["article_id"]
        cid = ref.get("clause_id", "")
        doc_str = chunk_by_article.get(aid)
        if not doc_str:
            continue
        answer_articles.add(aid)
        article_refs.add(format_article_ref(doc_str, aid, cid))

    for chunk in result.chunks:
        if not chunk.article_id:
            continue
        doc_str = format_doc_id(chunk)
        if not doc_str:
            continue
        if chunk.article_id not in answer_articles:
            article_refs.add(format_article_ref(doc_str, chunk.article_id))

    return {
        "id": qid,
        "question": question,
        "answer": result.final_answer,
        "relevant_docs": sorted(doc_refs),
        "relevant_articles": sorted(article_refs),
    }


async def run_submission(
    test_file: str,
    output_dir: str = "data/results",
    max_queries: int = 0,
    sources: list[str] | None = None,
    mode: str = "agentic",
):
    from src.data.loading import load_corpus
    from src.pipeline.orchestrator import LegalRAGPipeline

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_queries > 0:
        test_data = test_data[:max_queries]

    logger.info(f"Loading corpus...")
    docs = load_corpus(force_rebuild=False, sources=sources)
    if not docs:
        logger.error("No docs loaded!")
        return

    logger.info(f"Initializing pipeline with {len(docs)} docs...")
    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        use_decomposition=True,
    )

    use_agentic = mode == "agentic"
    logger.info(f"Running {len(test_data)} queries (mode={mode})...")
    entries = []
    for item in tqdm(test_data, desc="Processing"):
        qid = item["id"]
        question = item["question"]
        try:
            if use_agentic:
                result = await pipeline.answer_agentic(question)
            else:
                result = await pipeline.answer(question)
            entry = build_submission_entry(qid, question, result)
            entries.append(entry)
            logger.info(
                f"[{qid}] type={result.query_type} "
                f"docs={len(entry['relevant_docs'])} "
                f"articles={len(entry['relevant_articles'])}"
            )
        except Exception as e:
            logger.error(f"[{qid}] Error: {e}")
            entries.append({
                "id": qid,
                "question": question,
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": [],
            })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved results: {results_path}")

    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, "results.json")
    logger.info(f"Submission zip: {zip_path}")

    total_articles = sum(len(e["relevant_articles"]) for e in entries)
    avg_articles = total_articles / len(entries) if entries else 0
    logger.info(f"Summary: {len(entries)} entries, avg {avg_articles:.1f} articles/query")

    return entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R2AI2026 Legal Submission")
    parser.add_argument("--test-file", required=True, help="Path to test set JSON")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all")
    parser.add_argument("--mode", choices=["agentic", "oneshot"], default="agentic",
                        help="agentic = multi-round LLM-guided retrieval, oneshot = single-pass pipeline")
    args = parser.parse_args()

    asyncio.run(run_submission(
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        mode=args.mode,
    ))
