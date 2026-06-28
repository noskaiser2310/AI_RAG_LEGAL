import re

from src.core.base import ChunkStrategy

TIER_CONFIGS = {
    "article": ChunkStrategy(chunk_size=0, overlap=0, respect_boundaries=True, hierarchical=True),
    "semantic": ChunkStrategy(chunk_size=512, overlap=128, respect_boundaries=True, hierarchical=False),
    "long": ChunkStrategy(chunk_size=2048, overlap=256, respect_boundaries=True, hierarchical=False),
}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_by_legal_structure(text: str) -> list[tuple[str, str, str]]:
    try:
        sections = re.split(r"(?=Điều\s+\d+[\.\s])", text)
    except Exception:
        sections = [text]
    if len(sections) <= 1:
        try:
            sections = re.split(r"(?=Chương\s+\w+[\.\s])", text)
        except Exception:
            sections = [text]

    result = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"(Điều\s+\d+[^.]*?\.|Chương\s+\w+[^.]*?\.|Mục\s+\w+[^.]*?\.)", section)
        title = title_match.group(1).strip() if title_match else ""
        content = section[len(title):].strip() if title else section
        article_id = ""
        article_match = re.match(r"Điều\s+(\d+)", title)
        if article_match:
            article_id = article_match.group(1)
        result.append((title, article_id, content))
    if not result:
        result = [("", "", text)]
    return result


def chunk_text(
    text: str,
    doc_id: str,
    doc_title: str = "",
    url: str = "",
    strategy: ChunkStrategy | None = None,
) -> list[dict]:
    if strategy is None:
        strategy = TIER_CONFIGS["semantic"]
    chunks = []
    try:
        articles = split_by_legal_structure(text)
    except Exception:
        articles = [("", "", text)]

    if strategy.hierarchical and len(articles) > 1:
        for title, article_id, content in articles:
            words = content.split()
            if len(words) <= strategy.chunk_size * 1.5 or strategy.chunk_size == 0:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_art{article_id or '0'}",
                    "article_id": article_id,
                    "title": doc_title,
                    "section": title,
                    "text": content,
                    "url": url,
                })
            else:
                sub_chunks = _sliding_window_chunks(content, strategy)
                for j, sc in enumerate(sub_chunks):
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_art{article_id or '0'}_chunk{j}",
                        "article_id": article_id,
                        "title": doc_title,
                        "section": title,
                        "text": sc,
                        "url": url,
                    })
    else:
        sub_chunks = _sliding_window_chunks(text, strategy)
        for j, sc in enumerate(sub_chunks):
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk{j}",
                "article_id": "",
                "title": doc_title,
                "section": "",
                "text": sc,
                "url": url,
            })
    return chunks


def chunk_text_tiered(
    text: str,
    doc_id: str,
    doc_title: str = "",
    url: str = "",
) -> dict[str, list[dict]]:
    tiers = {}
    for tier_name, strategy in TIER_CONFIGS.items():
        chunks = chunk_text(text, doc_id, doc_title, url, strategy)
        for c in chunks:
            c["tier"] = tier_name
        tiers[tier_name] = chunks
    return tiers


def _sliding_window_chunks(text: str, strategy: ChunkStrategy) -> list[str]:
    words = text.split()
    if strategy.chunk_size == 0 or len(words) <= strategy.chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + strategy.chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if strategy.respect_boundaries and end < len(words):
            if not re.search(r"[.!?]\s*$", chunk):
                next_period = chunk.rfind(".")
                next_newline = chunk.rfind("\n")
                split_point = max(next_period, next_newline)
                if split_point > len(chunk) // 2:
                    end = len(" ".join(words[start:]).split()[: len(chunk[:split_point].split())])

        chunks.append(" ".join(words[start:end]))
        start = end - strategy.overlap
        if start < 0:
            start = 0
    return chunks
