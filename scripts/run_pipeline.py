import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    from src.data.loading import build_corpus
    from src.pipeline.orchestrator import LegalRAGPipeline
    from src.core.config import config

    logger.info("Building full corpus (all sources)...")
    docs = build_corpus(force_rebuild=True)
    logger.info(f"Corpus: {len(docs)} docs")

    logger.info(f"Initializing pipeline on {config.DEVICE.upper()}...")
    pipeline = await LegalRAGPipeline.create(
        docs=docs,
        dense_path=str(Path(config.INDEX_DIR) / "dense.index"),
        sparse_path=str(Path(config.INDEX_DIR) / "sparse"),
        use_decomposition=True,
    )

    test_questions = [
        "Người lao động có được hưởng lương trong thời gian thử việc không?",
        "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?",
        "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?",
        "Thời hiệu khởi kiện về hợp đồng dân sự là bao lâu?",
        "Trẻ em dưới 14 tuổi có được cấp Căn cước không?",
    ]

    logger.info(f"Running {len(test_questions)} test queries...")
    results = []
    for q in test_questions:
        t0 = time.time()
        try:
            result = await pipeline.answer(q)
            elapsed = time.time() - t0
            results.append({
                "query": q,
                "query_type": result.query_type,
                "answer": result.final_answer,
                "confidence": result.confidence,
                "citations": result.citations,
                "retrieval_time": result.retrieval_time,
                "generation_time": result.generation_time,
                "total_time": elapsed,
                "correction_rounds": result.num_correction_rounds,
            })
            logger.info(f"Q: {q[:50]}... conf={result.confidence:.3f} time={elapsed:.1f}s")
        except Exception as e:
            logger.error(f"Error: {q[:50]} -> {e}")
            results.append({"query": q, "error": str(e)})

    output_dir = Path(config.DATA_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {output_path}")

    for r in results:
        if "error" in r:
            print(f"\nERROR: {r['query']} -> {r['error']}")
            continue
        print(f"\n{'='*60}")
        print(f"Q: {r['query']}")
        print(f"Type: {r['query_type']}")
        print(f"A: {r['answer'][:300]}...")
        print(f"Conf: {r['confidence']:.4f} | Time: {r['total_time']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
