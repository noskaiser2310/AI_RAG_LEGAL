import re
import logging

logger = logging.getLogger(__name__)

# Pattern: Điều followed by number, then period or colon
ARTICLE_PATTERN = re.compile(r'^Điều\s+(\d+)\s*[\.\:]\s*(.*)', re.MULTILINE)

# Pattern to split on article boundaries (newline + "Điều X.")
SPLIT_PATTERN = re.compile(r'\n(?=Điều\s+\d+\s*[\.\:])')

# For detecting "Căn cứ" preamble end
PREAMBLE_END = re.compile(r'\n(QUYẾT ĐỊNH|QUYẾT NGHỊ|NGHỊ QUYẾT|NGHỊ ĐỊNH|THÔNG TƯ|THÔNG BÁO|Điều\s+\d+\s*[\.\:])\s*[:：]?\s*\n?')


def extract_title_from_text(text: str) -> str:
    """Extract document title from the preamble section."""
    lines = text.strip().split('\n')
    # Title is usually after the document type line, before "Căn cứ"
    for i, line in enumerate(lines):
        line = line.strip()
        # Skip header lines
        if 'CỘNG HÒA' in line or 'Số:' in line or 'Căn cứ' in line or not line:
            continue
        # If it's all caps and > 20 chars, it's likely the title
        if line.isupper() and len(line) > 20:
            return line
    return ''


def extract_articles_from_markdown(
    doc_id: int,
    title: str,
    content: str,
    doc_type: str = '',
    doc_number: str = '',
    issuing_authority: str = '',
    issuance_date: str = '',
    source: str = 'vohuutridung',
    min_article_length: int = 20,
) -> list[dict]:
    """
    Extract individual articles from a legal document in Markdown format.
    
    Returns list of dicts with keys:
        text, title, doc_id, article_id, source, doc_type, doc_number,
        issuing_authority, issuance_date
    """
    # Split content into preamble and articles
    parts = SPLIT_PATTERN.split(content.strip())
    
    docs = []
    article_counter = 0
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        
        # Check if this part starts with an article
        m = ARTICLE_PATTERN.match(part)
        if m:
            article_num = m.group(1)
            article_title = m.group(2).strip()
            # Get first line of article as title if available
            first_line = article_title.split('\n')[0].strip()
            # Remove clause numbers like "1." from title
            first_line = re.sub(r'^\d+\.\s*', '', first_line)
            
            article_counter += 1
            
            doc = {
                'text': part,
                'title': title if not first_line else f'{title} - Điều {article_num}',
                'article_title': first_line,
                'doc_id': f'tvpl_{doc_id}',
                'article_id': f'Điều {article_num}',
                'source': source,
                'doc_type': doc_type,
                'doc_number': doc_number,
                'issuing_authority': issuing_authority,
                'issuance_date': issuance_date,
                'hierarchy': f'{doc_number}/{article_num}' if doc_number else '',
            }
            docs.append(doc)
        else:
            # This is the preamble (before first article)
            # We still create a document for the preamble to handle
            # documents that may not have articles (administrative docs)
            if i == 0 and not parts[i+1:i+2]:
                # Only document-level content, no articles
                first_line = part.split('\n')[0].strip() if part else ''
                docs.append({
                    'text': part,
                    'title': title,
                    'article_title': '',
                    'doc_id': f'tvpl_{doc_id}',
                    'article_id': '',
                    'source': source,
                    'doc_type': doc_type,
                    'doc_number': doc_number,
                    'issuing_authority': issuing_authority,
                    'issuance_date': issuance_date,
                    'hierarchy': doc_number or '',
                })
    
    # Filter by minimum length
    docs = [d for d in docs if len(d['text']) >= min_article_length]
    
    if not docs:
        # Fallback: if no articles extracted, create a single document
        docs.append({
            'text': content,
            'title': title,
            'article_title': '',
            'doc_id': f'tvpl_{doc_id}',
            'article_id': '',
            'source': source,
            'doc_type': doc_type,
            'doc_number': doc_number,
            'issuing_authority': issuing_authority,
            'issuance_date': issuance_date,
            'hierarchy': doc_number or '',
        })
    
    return docs


def _cached_hf_path(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, repo_type=repo_type, local_files_only=True)
    except (OSError, RuntimeError):
        return hf_hub_download(repo_id, filename, repo_type=repo_type)


# Known parquet files for vohuutridung — hardcoded to skip list_repo_tree() API calls.
VOHUUTRIDUNG_METADATA_FILES = ["metadata/data-00000-of-00001.parquet"]
VOHUUTRIDUNG_CONTENT_FILES = [f"content/data-{i:05d}-of-00011.parquet" for i in range(11)]


def extract_all_vohuutridung(
    max_docs: int | None = None,
    article_only: bool = True,
    min_article_length: int = 20,
    filter_types: list[str] | None = None,
    output_file: str | None = None,
) -> list[dict]:
    """
    Load vohuutridung dataset via direct parquet download (avoids PyArrow streaming crash),
    extract articles, return flat list of documents.
    Uses duckdb for streaming parquet reads (low memory, avoids pyarrow realloc issues).
    If output_file is provided, writes articles to JSONL incrementally (avoids MemoryError).
    """
    import json as _json
    import duckdb
    import pandas as pd

    REPO = "vohuutridung/vietnamese-legal-documents"

    # --- Load metadata ---
    logger.info("Loading vohuutridung metadata (direct parquet)...")
    meta_map = {}
    for fname in VOHUUTRIDUNG_METADATA_FILES:
        path = _cached_hf_path(REPO, fname)
        df = pd.read_parquet(path)
        id_col = "id" if "id" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            meta_map[row[id_col]] = row.to_dict()
        del df
        if max_docs and len(meta_map) >= max_docs:
            break
    logger.info(f"Loaded {len(meta_map)} metadata entries")

    if filter_types:
        meta_map = {
            k: v for k, v in meta_map.items()
            if v.get('legal_type', '') in filter_types
        }
        logger.info(f"Filtered to {len(meta_map)} docs of types: {filter_types}")
    else:
        from src.data.filtering import TIER_KEEP
        before = len(meta_map)
        meta_map = {
            k: v for k, v in meta_map.items()
            if v.get('legal_type', '') in TIER_KEEP
        }
        after = len(meta_map)
        logger.info(f"Smart filter: {before} -> {after} docs (dropped {before-after} TIER_DROP)")

    # --- Load content (via duckdb for memory-efficient streaming) ---
    logger.info("Loading vohuutridung content (duckdb streaming)...")
    content_files = VOHUUTRIDUNG_CONTENT_FILES

    all_articles = []
    out_fh = None
    total_written = 0
    if output_file:
        out_fh = open(output_file, "a", encoding="utf-8")
        logger.info(f"  Writing articles incrementally to {output_file}")

    skipped_no_meta = 0
    skipped_no_articles = 0
    processed = 0
    BATCH_SIZE = 500

    for fname in content_files:
        path = _cached_hf_path(REPO, fname)
        conn = duckdb.connect()
        pq_path = path.replace("\\", "/")
        total = conn.execute(f"SELECT count(*) FROM read_parquet('{pq_path}')").fetchone()[0]
        col_info = conn.execute(f"SELECT * FROM read_parquet('{pq_path}') LIMIT 0").description
        col_names = [c[0] for c in col_info]
        id_col = "id" if "id" in col_names else col_names[0]
        content_col = next((c for c in ("content", "content_text", "text") if c in col_names), None)
        if not content_col:
            logger.warning(f"  No content column in {fname}")
            conn.close()
            continue

        file_articles = 0
        for offset in range(0, total, BATCH_SIZE):
            rows = conn.execute(
                f"SELECT {id_col}, {content_col} FROM read_parquet('{pq_path}') LIMIT {BATCH_SIZE} OFFSET {offset}"
            ).fetchall()
            for doc_id, content in rows:
                processed += 1
                content = str(content) if content else ""

                if not content or len(content.strip()) < 30:
                    continue

                meta = meta_map.get(doc_id, {})
                if not meta:
                    skipped_no_meta += 1
                    if filter_types is not None:
                        continue

                title = meta.get('title', '') or f'VB {doc_id}'
                doc_type = meta.get('legal_type', '') or ''
                doc_number = meta.get('document_number', '') or ''
                issuing_authority = meta.get('issuing_authority', '') or ''
                issuance_date = meta.get('issuance_date', '') or ''

                articles = extract_articles_from_markdown(
                    doc_id=doc_id if isinstance(doc_id, int) else hash(str(doc_id)) % 100000,
                    title=title,
                    content=content,
                    doc_type=doc_type,
                    doc_number=doc_number,
                    issuing_authority=issuing_authority,
                    issuance_date=issuance_date,
                    source='vohuutridung',
                    min_article_length=min_article_length,
                )

                if article_only:
                    has_articles = any(a.get('article_id') for a in articles)
                    if not has_articles:
                        skipped_no_articles += 1
                        continue

                if out_fh:
                    for art in articles:
                        if not art.get('so_ky_hieu'):
                            dn = art.get('doc_number', '')
                            if dn:
                                art['so_ky_hieu'] = dn
                            else:
                                from src.retrieval.text_processor import extract_doc_code as _edc
                                code = _edc(art.get('title', ''))
                                if code:
                                    art['so_ky_hieu'] = code
                        out_fh.write(_json.dumps(art, ensure_ascii=False) + "\n")
                    total_written += len(articles)
                else:
                    all_articles.extend(articles)
                file_articles += len(articles)

                if max_docs and processed >= max_docs:
                    break

            if max_docs and processed >= max_docs:
                break

        conn.close()
        logger.info(f"  {fname}: {file_articles} articles ({processed} rows total)")
        if max_docs and processed >= max_docs:
            break

    if out_fh:
        out_fh.close()
        logger.info(f"Total articles written to file: {total_written} "
                    f"(no_meta={skipped_no_meta}, no_articles={skipped_no_articles})")
        return []

    logger.info(f"Total articles: {len(all_articles)} "
                f"(no_meta={skipped_no_meta}, no_articles={skipped_no_articles})")

    return all_articles
