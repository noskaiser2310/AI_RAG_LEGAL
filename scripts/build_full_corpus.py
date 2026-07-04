"""
Build full corpus from all sources.
Loads each source independently and saves consolidated JSONL.
"""
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_source(source_name: str, limit: int | None) -> list[dict]:
    from src.data.loading import SOURCE_LOADERS
    loader = SOURCE_LOADERS.get(source_name)
    if loader is None:
        raise ValueError(f"Unknown source: {source_name}")
    return loader(limit)


def main():
    t0 = time.time()

    PER_SOURCE = {
        "phapdien": None,
        "vohuutridung": None,   # Smart filter drops ~80% docs
        "th1nhng0": None,
        "utslvc": None,
        "kiencute": 5000,       # Pre-training corpus (noisy, limit)
        "pbgdpl": None,
    }

    all_docs = []
    for source_name, limit in PER_SOURCE.items():
        s0 = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Loading {source_name} (limit={limit})...")
        try:
            docs = load_source(source_name, limit)
            all_docs.extend(docs)
            logger.info(f"  -> {len(docs)} docs in {time.time()-s0:.1f}s")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    # Enrich with doc_code
    from src.retrieval.text_processor import extract_doc_code
    enriched = 0
    for doc in all_docs:
        if not doc.get("so_ky_hieu"):
            code = extract_doc_code(doc.get("title", ""))
            if code:
                doc["so_ky_hieu"] = code
                enriched += 1
    logger.info(f"Enriched {enriched}/{len(all_docs)} docs with so_ky_hieu")

    # Save
    corpus_path = Path("data") / "processed" / "corpus_full.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    src_counts = Counter(d.get('source', '?') for d in all_docs)
    art_counts = Counter(
        'has_article' if d.get('article_id') else 'no_article'
        for d in all_docs
    )

    print(f"\n{'='*60}")
    print(f"CORPUS BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(all_docs)}")
    print(f"By source: {dict(src_counts)}")
    print(f"Article coverage: {dict(art_counts)}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Saved: {corpus_path}")

    stats = {
        "total_docs": len(all_docs),
        "by_source": dict(src_counts),
        "by_article": dict(art_counts),
        "build_time_seconds": elapsed,
    }
    with open(corpus_path.parent / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
