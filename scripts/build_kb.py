"""Build corpus + dense/sparse indexes from all data sources.

Usage:
    python scripts/build_kb.py                          # full build (corpus + indexes)
    python scripts/build_kb.py --rebuild                # force rebuild from scratch
    python scripts/build_kb.py --index-only             # skip corpus build, only indexes
    python scripts/build_kb.py --skip-corpus            # same as --index-only
    python scripts/build_kb.py --max-per-source 100     # test with small subset
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("TRANSFORMERS_CACHE", str(_ROOT / "hf_cache"))
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ.setdefault("EMBEDDING_MODEL", "mainguyen9/vietlegal-harrier-0.6b")
os.environ.setdefault("EMBEDDING_DIM", "1024")
os.environ.setdefault("DEVICE", "cuda")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_kb")


async def main(
    skip_corpus: bool = False,
    rebuild: bool = False,
    max_per_source: int | None = None,
    batch_size: int = 64,
):
    from src.core.config import config
    from src.data.loading import load_corpus
    from src.embedding.harrier_embedding import HarrierEmbedding
    from src.retrieval.indexing import DenseIndex, SparseIndex
    from src.retrieval.text_processor import prepare_for_bm25

    data_dir = Path(config.DATA_DIR)
    index_dir = Path(config.INDEX_DIR)
    corpus_path = data_dir / "processed" / config.CORPUS_FILE
    dense_path = index_dir / "dense.index"
    sparse_path = index_dir / "sparse"

    index_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Step 1: Build or load corpus ──
    if rebuild:
        logger.info("Force rebuild, removing old corpus and indexes...")
        for p in [corpus_path, dense_path, sparse_path.with_suffix(".meta.json")]:
            if p.exists():
                if p.is_dir():
                    import shutil; shutil.rmtree(p)
                else:
                    p.unlink()
                logger.info(f"  Removed: {p}")
        docs = load_corpus(force_rebuild=True, sources=None)
    elif skip_corpus:
        logger.info("Skipping corpus build, loading existing...")
        docs = load_corpus(force_rebuild=False)
    elif corpus_path.exists():
        logger.info(f"Corpus exists, loading: {corpus_path}")
        docs = load_corpus(force_rebuild=False)
    else:
        logger.info("=" * 60)
        logger.info("STEP 1: Building corpus from all sources")
        logger.info("=" * 60)
        if dense_path.exists():
            dense_path.unlink()
            logger.info(f"Removed old dense index: {dense_path}")
        if sparse_path.with_suffix(".meta.json").exists():
            sparse_path.with_suffix(".meta.json").unlink()
            logger.info(f"Removed old sparse index: {sparse_path}")

        docs = load_corpus(force_rebuild=True, sources=None)

    if not docs:
        logger.error("No docs loaded! Aborting.")
        return

    logger.info(f"Corpus: {len(docs)} docs from {corpus_path}")

    # ── Step 2: Build dense index ──
    logger.info("=" * 60)
    logger.info("STEP 2: Building dense index (HarrierEmbedding 0.6B)")
    logger.info("=" * 60)

    if dense_path.exists():
        logger.info(f"Dense index exists, loading: {dense_path}")
        dense_idx = DenseIndex()
        dense_idx.load(dense_path)
    else:
        logger.info(f"Encoding {len(docs)} docs with HarrierEmbedding...")
        embedder = HarrierEmbedding()
        dense_idx = DenseIndex()

        texts = [d["text"] for d in docs]
        ids_list = list(range(len(docs)))
        all_embs = []

        import numpy as np
        from tqdm import tqdm

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            embs = await embedder.embed(batch)
            all_embs.extend(embs)

        emb_array = np.array(all_embs)
        id_array = np.array(ids_list, dtype=np.int64)
        dense_idx.add(emb_array, id_array)
        dense_idx.save(dense_path)
        logger.info(f"Dense index saved: {dense_path} ({dense_idx.index.ntotal} vectors)")
        del embedder
        import torch
        torch.cuda.empty_cache()

    t1 = time.time()
    logger.info(f"Dense index: {t1 - t0:.0f}s")

    # ── Step 3: Build sparse index (BM25) ──
    logger.info("=" * 60)
    logger.info("STEP 3: Building sparse index (BM25)")
    logger.info("=" * 60)

    if sparse_path.with_suffix(".meta.json").exists():
        logger.info(f"Sparse index exists, loading...")
        sparse_idx = SparseIndex()
        sparse_idx.load(sparse_path)
    else:
        texts = [d["text"] for d in docs]
        ids_list = list(range(len(docs)))
        sparse_idx = SparseIndex(use_vn_segmentation=config.USE_VN_SEGMENTATION)
        sparse_idx.build(texts, ids_list)
        sparse_idx.save(sparse_path)
        logger.info(f"Sparse index saved: {sparse_path} (bm25s)")

    t2 = time.time()
    logger.info(f"Sparse index: {t2 - t1:.0f}s")

    # ── Summary ──
    total = time.time() - t0
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Corpus:       {len(docs)} docs -> {corpus_path}")
    logger.info(f"  Dense index:  {dense_idx.index.ntotal} vectors -> {dense_path}")
    logger.info(f"  Sparse index: {len(sparse_idx.corpus)} docs -> {sparse_path} (bm25s)")
    logger.info(f"  Total time:   {total:.0f}s ({total/60:.1f} min)")
    logger.info(f"  Project root: {_ROOT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LegalRAG KB + indexes")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild corpus + indexes from scratch")
    parser.add_argument("--skip-corpus", action="store_true", help="Skip corpus rebuild (alias: --index-only)")
    parser.add_argument("--index-only", action="store_true", help="Skip corpus rebuild, only build indexes")
    parser.add_argument("--max-per-source", type=int, default=None, help="Limit docs per source (for testing)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    asyncio.run(main(
        skip_corpus=args.skip_corpus or args.index_only,
        rebuild=args.rebuild,
        max_per_source=args.max_per_source,
        batch_size=args.batch_size,
    ))
