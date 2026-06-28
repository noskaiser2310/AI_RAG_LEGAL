import json
import logging
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.core.config import config
from src.data.article_extractor import extract_articles_from_markdown
from src.retrieval.text_processor import extract_doc_code

logger = logging.getLogger(__name__)


DATA_SOURCES = {
    "phapdien": {
        "name": "Pháp Điển",
        "hf_path": "tmquan/phapdien-moj-gov-vn",
        "config": "articles",
        "split": "train",
        "priority": 1,
    },
    "vohuutridung": {
        "name": "Thư Viện PL",
        "hf_path": "vohuutridung/vietnamese-legal-documents",
        "config": "content",
        "split": "data",
        "priority": 2,
    },
    "th1nhng0": {
        "name": "VBPL (legacy)",
        "hf_path": "th1nhng0/vietnamese-legal-documents",
        "config": "metadata",
        "split": "data",
        "priority": 3,
    },
    "utslvc": {
        "name": "UTS_VLC",
        "hf_path": "undertheseanlp/UTS_VLC",
        "config": None,
        "split": "2026",
        "priority": 4,
    },
    "kiencute": {
        "name": "Legal Pretrain",
        "hf_path": "KienCute/legal-pretrain",
        "config": None,
        "split": "train",
        "priority": 5,
    },
    "pbgdpl": {
        "name": "PBGDPL Q&A",
        "hf_path": "tmquan/pbgdpl-vn-legal-qna",
        "config": None,
        "split": "train",
        "priority": 6,
    },
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _cached_hf_path(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, repo_type=repo_type, local_files_only=True)
    except (OSError, RuntimeError):
        return hf_hub_download(repo_id, filename, repo_type=repo_type)


PHAPDIEN_ARTICLE_FILES = [f"articles-{i:05d}-of-00007.parquet" for i in range(7)]


def load_phapdien_articles(max_articles: int | None = None) -> list[dict]:
    """Load Pháp Điển articles via direct parquet download (avoids PyArrow streaming crash on Windows)."""
    import pyarrow.parquet as pq

    logger.info("Loading Pháp Điển (direct parquet)...")
    REPO = "tmquan/phapdien-moj-gov-vn"
    docs = []
    for fname in PHAPDIEN_ARTICLE_FILES:
        path = _cached_hf_path(REPO, fname)
        table = pq.read_table(path)
        cols = table.column_names
        text_col = next((c for c in ("content_text", "text", "content") if c in cols), None)
        if not text_col:
            logger.warning(f"  No text column in {fname}, columns={cols}")
            continue
        art_col = next((c for c in ("article_id",) if c in cols), None)
        title_col = next((c for c in ("article_title",) if c in cols), None)
        rid_col = next((c for c in ("record_id", "id") if c in cols), None)
        src_col = next((c for c in ("source_note_text",) if c in cols), None)
        sid_col = next((c for c in ("subject_id",) if c in cols), None)
        for i in range(len(table)):
            text = str(table[text_col][i].as_py() or "")
            if not text or len(text.strip()) < 20:
                continue
            raw_article_id = str(table[art_col][i].as_py() or "") if art_col else ""
            article_title = str(table[title_col][i].as_py() or "") if title_col else ""
            record_id = str(table[rid_col][i].as_py() or "") if rid_col else ""
            source_note = str(table[src_col][i].as_py() or "") if src_col else ""
            subject_id = str(table[sid_col][i].as_py() or "") if sid_col else ""

            so_ky_hieu = ""
            doc_type = ""
            m_code = re.search(r"số\s+(\d+/\d+/[\w-]+)", source_note)
            if m_code:
                so_ky_hieu = m_code.group(1)
            m_type = re.search(r"(Luật|Bộ luật|Nghị định|Thông tư|Pháp lệnh|Nghị quyết|Hiến pháp)", source_note)
            if m_type:
                doc_type = m_type.group(1)

            article_id = raw_article_id
            if raw_article_id.startswith("Điều "):
                compound = raw_article_id[5:]
                parts = compound.split(".")
                if len(parts) >= 2:
                    if not so_ky_hieu:
                        so_ky_hieu = ".".join(parts[:-1])
                    article_id = parts[-1]
                else:
                    article_id = compound
            elif "." in raw_article_id:
                parts = raw_article_id.split(".")
                if len(parts) >= 2:
                    if not so_ky_hieu:
                        so_ky_hieu = ".".join(parts[:-1])
                    article_id = parts[-1]

            if article_title.startswith("Điều "):
                dot_pos = article_title.find(". ", 5)
                if dot_pos > 0:
                    article_title = article_title[dot_pos + 2:]

            if doc_type and so_ky_hieu:
                title = f"{doc_type} {so_ky_hieu}"
            elif so_ky_hieu:
                title = so_ky_hieu
            else:
                title = article_title or f"Điều {article_id}"

            docs.append({
                "text": clean_text(text),
                "title": title,
                "doc_id": f"phapdien_{subject_id or record_id}",
                "article_id": article_id,
                "so_ky_hieu": so_ky_hieu,
                "source": "phapdien",
            })
            if max_articles and len(docs) >= max_articles:
                break
        logger.info(f"  {fname}: {len(docs)} articles so far")
        if max_articles and len(docs) >= max_articles:
            break
    logger.info(f"  {len(docs)} articles total")
    return docs


def load_vohuutridung_articles(max_docs: int | None = None) -> list[dict]:
    logger.info("Loading vohuutridung with article extraction...")
    from src.data.article_extractor import extract_all_vohuutridung
    return extract_all_vohuutridung(max_docs=max_docs, article_only=False)


def load_th1nhng0_articles(max_docs: int | None = None) -> list[dict]:
    logger.info("Loading th1nhng0 with article extraction...")
    from src.data.th1nhng0_loader import load_th1nhng0_with_articles
    return load_th1nhng0_with_articles(max_docs=max_docs, article_only=False)


def load_utslvc_docs(max_docs: int | None = None) -> list[dict]:
    logger.info("Loading UTS_VLC...")
    ds = load_dataset("undertheseanlp/UTS_VLC", split="2026", streaming=False)
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    docs = []
    for row in tqdm(ds, desc="UTS_VLC"):
        content = row.get("content", "")
        title = row.get("title", "")
        if not content or len(str(content).strip()) < 20:
            continue
        docs.append({
            "text": clean_text(str(content)),
            "title": title or f"VB {row.get('id', '')}",
            "doc_id": f"utslvc_{row.get('id', '')}",
            "article_id": "",
            "source": "utslvc",
        })
    logger.info(f"  {len(docs)} laws")
    return docs


def load_kiencute_pretrain(max_docs: int | None = None) -> list[dict]:
    logger.info("Loading KienCute/legal-pretrain...")
    ds = load_dataset("KienCute/legal-pretrain", split="train", streaming=False)
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    docs = []
    count = 0
    for row in tqdm(ds, desc="KienCute"):
        doc_content = row.get("doc_content", "")
        meta = row.get("metadata", {}) or {}
        if not doc_content or len(doc_content.strip()) < 30:
            continue
        if len(doc_content) > 500_000:
            continue
        try:
            text = clean_text(doc_content)
        except MemoryError:
            continue
        docs.append({
            "text": text,
            "title": meta.get("DocName", "") or f"VB {meta.get('Id', count)}",
            "doc_id": f"kiencute_{meta.get('Id', count)}",
            "article_id": "",
            "issuing_authority": meta.get("OrganName", ""),
            "source": "kiencute",
        })
        count += 1
    logger.info(f"  {len(docs)} docs")
    return docs


def load_pbgdpl_qa(max_qa: int | None = None) -> list[dict]:
    logger.info("Loading PBGDPL Q&A...")
    ds = load_dataset("tmquan/pbgdpl-vn-legal-qna", split="train", streaming=False)
    if max_qa:
        ds = ds.select(range(min(max_qa, len(ds))))
    docs = []
    for row in tqdm(ds, desc="PBGDPL"):
        question = row.get("question_text", row.get("question", ""))
        answer = row.get("answer_text", row.get("answer", ""))
        if not answer or len(str(answer).strip()) < 20:
            continue
        docs.append({
            "text": clean_text(str(answer)),
            "title": str(question)[:200] if question else f"QA {row.get('item_id', '')}",
            "doc_id": f"pbgdpl_{row.get('item_id', '')}",
            "article_id": "",
            "source": "pbgdpl",
            "question": str(question) if question else "",
        })
    logger.info(f"  {len(docs)} Q&A")
    return docs


SOURCE_LOADERS = {
    "phapdien": load_phapdien_articles,
    "vohuutridung": load_vohuutridung_articles,
    "th1nhng0": load_th1nhng0_articles,
    "utslvc": load_utslvc_docs,
    "kiencute": load_kiencute_pretrain,
    "pbgdpl": load_pbgdpl_qa,
}


def _chunk_docs(docs: list[dict]) -> list[dict]:
    from src.retrieval.chunking import split_by_legal_structure

    chunked = []
    split_count = 0
    for doc in docs:
        if doc.get("article_id"):
            chunked.append(doc)
            continue
        text = doc.get("text", "")
        if len(text) < 1800:
            chunked.append(doc)
            continue
        sections = split_by_legal_structure(text)
        if len(sections) <= 1:
            chunked.append(doc)
            continue
        split_count += 1
        for i, (title, article_id, content) in enumerate(sections):
            if not content or len(content.strip()) < 20:
                continue
            sub = dict(doc)
            sub["text"] = content
            sub["article_id"] = article_id
            sub["chunk_id"] = f"{doc.get('doc_id', '')}_art{article_id or i}"
            if title:
                sub["title"] = doc.get("title", "") + f" - {title}"
            if not sub.get("so_ky_hieu"):
                sub["so_ky_hieu"] = extract_doc_code(sub.get("title", ""))
            chunked.append(sub)
    logger.info(f"  Chunked: {len(docs)} -> {len(chunked)} (split {split_count})")
    return chunked


def _enrich_docs(docs: list[dict]) -> int:
    enriched = 0
    for doc in docs:
        if not doc.get("so_ky_hieu"):
            doc_number = doc.get("doc_number", "")
            if doc_number:
                doc["so_ky_hieu"] = doc_number
                enriched += 1
            else:
                code = extract_doc_code(doc.get("title", ""))
                if code:
                    doc["so_ky_hieu"] = code
                    enriched += 1
    return enriched


def build_corpus(
    output_path: str | Path | None = None,
    max_samples_per_source: int | None = None,
    sources: list[str] | None = None,
    force_rebuild: bool = False,
) -> list[dict]:
    import gc

    output_path = Path(output_path or f"{config.DATA_DIR}/processed/{config.CORPUS_FILE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Corpus exists: {output_path}, loading cache")
        return load_corpus(cache_path=output_path)

    if sources is None:
        sources = ["phapdien", "vohuutridung", "th1nhng0", "utslvc", "kiencute", "pbgdpl"]

    temp_path = output_path.with_suffix(".tmp.jsonl")
    total_written = 0

    for key in sources:
        if key not in SOURCE_LOADERS:
            logger.warning(f"Unknown source: {key}")
            continue
        try:
            if key == "vohuutridung":
                from src.data.article_extractor import extract_all_vohuutridung
                logger.info("Loading source: vohuutridung (streaming to file)...")
                from src.data.filtering import TIER_KEEP
                extract_all_vohuutridung(
                    max_docs=max_samples_per_source,
                    article_only=True,
                    min_article_length=100,
                    filter_types=list(TIER_KEEP),
                    output_file=str(temp_path),
                )
                count = 0
                if temp_path.exists():
                    with open(temp_path, "r", encoding="utf-8") as fh:
                        for _ in fh:
                            count += 1
                total_written = count
                logger.info(f"  vohuutridung: temp file now has {count} lines total")
                gc.collect()
                continue

            logger.info(f"Loading source: {key}...")
            raw = SOURCE_LOADERS[key](max_samples_per_source)
            logger.info(f"  Loaded {len(raw)} docs from {key}")

            enriched = _enrich_docs(raw)
            logger.info(f"  Enriched {enriched}/{len(raw)} with so_ky_hieu")

            chunked = _chunk_docs(raw)
            del raw
            gc.collect()

            with open(temp_path, "a", encoding="utf-8") as f:
                for doc in chunked:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_written += len(chunked)
            logger.info(f"  Written {len(chunked)} docs (total: {total_written})")
            del chunked
            gc.collect()
        except Exception as e:
            logger.warning(f"  Error loading {key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if total_written == 0:
        logger.error("No data loaded!")
        return []

    logger.info(f"Total raw docs written: {total_written}")
    logger.info("Dedup + quality filter pass (streaming)...")

    seen_keys = set()
    kept = 0
    removed_dup = 0
    removed_short = 0

    with open(temp_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)

            aid = doc.get("article_id", "")
            if isinstance(aid, str) and aid.startswith("Điều "):
                doc["article_id"] = aid[5:].strip()

            text = doc.get("text", "")
            word_count = len(text.split())
            if word_count < 30:
                removed_short += 1
                continue

            code = doc.get("so_ky_hieu", "")
            art = doc.get("article_id", "")
            if code and art:
                dedup_key = f"{code}||{art}"
            else:
                cid = doc.get("chunk_id", "")
                dedup_key = cid if cid else f"_nokey_{kept}"

            if dedup_key in seen_keys:
                removed_dup += 1
                continue
            seen_keys.add(dedup_key)

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(f"Dedup: removed {removed_dup} duplicates, {removed_short} short/overflow")
    logger.info(f"Final corpus: {kept} docs")

    temp_path.unlink(missing_ok=True)
    logger.info(f"Saved {kept} docs to {output_path}")

    docs = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def load_corpus(
    cache_path: str | Path | None = None,
    max_chunks: int | None = None,
    sources: list[str] | None = None,
    force_rebuild: bool = False,
) -> list[dict]:
    cache_path = Path(cache_path or f"{config.DATA_DIR}/processed/{config.CORPUS_FILE}")

    if force_rebuild or not cache_path.exists():
        return build_corpus(output_path=cache_path, sources=sources, force_rebuild=force_rebuild)

    docs = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    total = len(docs)
    if max_chunks and max_chunks > 0:
        docs = docs[:max_chunks]
    logger.info(f"Loaded {len(docs)}/{total} docs from {cache_path}")
    return docs
