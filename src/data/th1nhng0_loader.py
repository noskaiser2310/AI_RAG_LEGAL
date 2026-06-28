"""
Fix for th1nhng0/vietnamese-legal-documents content loading.
The datasets library's Arrow schema inference tries to cast large_string -> string,
which fails for large documents. We bypass this by reading parquet directly.
"""
import logging
import os

logger = logging.getLogger(__name__)

HF_HOME = os.environ.get("HF_HOME", "") or ""


def load_th1nhng0_metadata(max_docs: int | None = None) -> list[dict]:
    """Load th1nhng0 metadata. Using streaming=False for stability."""
    from datasets import load_dataset
    from tqdm import tqdm

    logger.info("Loading th1nhng0 metadata...")
    ds = load_dataset('th1nhng0/vietnamese-legal-documents', 'metadata', split='data', streaming=False)
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))

    docs = []
    for row in tqdm(ds, desc='th1nhng0 metadata'):
        docs.append(dict(row))

    logger.info(f"Loaded {len(docs)} metadata entries")
    return docs


def _cached_hf_path(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, repo_type=repo_type, local_files_only=True)
    except (OSError, RuntimeError):
        return hf_hub_download(repo_id, filename, repo_type=repo_type)


def load_th1nhng0_content_direct(max_docs: int | None = None) -> list[dict]:
    """
    Load th1nhng0 content using pyarrow directly, bypassing the
    large_string -> string casting issue.
    """
    import pyarrow.parquet as pq

    logger.info("Loading th1nhng0 content (pyarrow direct)...")

    REPO = "th1nhng0/vietnamese-legal-documents"
    PARQUET_PATH = "data/content.parquet"

    path = _cached_hf_path(REPO, PARQUET_PATH)
    logger.info(f"Parquet file: {path}")

    # Read without schema -> keeps large_string as-is
    table = pq.read_table(path)
    logger.info(f"Rows: {len(table)}, schema: {table.schema}")

    docs = []
    for i in range(len(table)):
        doc = {
            'id': str(table['id'][i].as_py()),
            'content_html': str(table['content_html'][i].as_py()),
        }
        docs.append(doc)
        if max_docs and len(docs) >= max_docs:
            break

    logger.info(f"Loaded {len(docs)} content entries")
    return docs


def load_th1nhng0_relationships(max_docs: int | None = None) -> list[dict]:
    """Load th1nhng0 relationships config."""
    from datasets import load_dataset
    from tqdm import tqdm

    logger.info("Loading th1nhng0 relationships...")
    ds = load_dataset('th1nhng0/vietnamese-legal-documents', 'relationships', split='data', streaming=False)
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))

    docs = []
    for row in tqdm(ds, desc='th1nhng0 relationships'):
        docs.append(dict(row))

    logger.info(f"Loaded {len(docs)} relationships")
    return docs


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML content (th1nhng0 format)."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator='\n')

    # Normalize: fix "Điều X\n:" -> "Điều X:"
    text = re.sub(r'(Điều\s+\d+)\s*\n\s*([\.\:])', r'\1\2', text)
    # Remove non-breaking spaces and extra whitespace
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def load_th1nhng0_with_articles(
    max_docs: int | None = None,
    article_only: bool = True,
) -> list[dict]:
    """
    Load th1nhng0 content, extract articles from HTML, return flat list.
    """
    from src.data.article_extractor import extract_articles_from_markdown

    # Load metadata
    metadata = load_th1nhng0_metadata(max_docs=max_docs)
    meta_map = {m['id']: m for m in metadata}

    # Load content directly
    content = load_th1nhng0_content_direct(max_docs=max_docs)

    from tqdm import tqdm

    all_articles = []
    skipped_no_meta = 0
    skipped_no_articles = 0

    for item in tqdm(content, desc='Extracting articles (th1nhng0)'):
        doc_id = item['id']
        html = item['content_html']
        text = extract_text_from_html(html)

        if not text or len(text) < 30:
            continue

        meta = meta_map.get(int(doc_id)) if doc_id.isdigit() else meta_map.get(doc_id)
        if not meta:
            skipped_no_meta += 1
            continue

        title = meta.get('title', '') or f'VB {doc_id}'
        doc_type = meta.get('loai_van_ban', '') or ''
        doc_number = meta.get('so_ky_hieu', '') or ''

        articles = extract_articles_from_markdown(
            doc_id=int(doc_id) if doc_id.isdigit() else 0,
            title=title,
            content=text,
            doc_type=doc_type,
            doc_number=doc_number,
            issuing_authority=meta.get('co_quan_ban_hanh', '') or '',
            issuance_date=meta.get('ngay_ban_hanh', '') or '',
            source='th1nhng0',
        )

        if article_only:
            has_articles = any(a.get('article_id') for a in articles)
            if not has_articles:
                skipped_no_articles += 1
                continue

        all_articles.extend(articles)

    logger.info(f"Total articles from th1nhng0: {len(all_articles)} "
                f"(no_meta={skipped_no_meta}, no_articles={skipped_no_articles})")

    return all_articles
