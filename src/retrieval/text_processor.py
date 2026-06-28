import re
import unicodedata

VIETNAMESE_STOPWORDS = {
    "và", "của", "các", "có", "được", "cho", "trong", "với", "không",
    "là", "đến", "tại", "theo", "sau", "khi", "trên", "như", "này",
    "việc", "bị", "đã", "sẽ", "đang", "để", "về", "từ", "hoặc",
    "thì", "mà", "do", "nếu", "vì", "nên", "ra", "lại", "đó",
    "những", "một", "người", "cần", "phải", "cũng", "hay", "qua",
}

LEGAL_TERM_MAP = {
    "ld": "lao động",
    "lđ": "lao động",
    "blđ": "bộ luật lao động",
    "bl": "bộ luật",
    "nd": "nghị định",
    "tt": "thông tư",
    "cp": "chính phủ",
    "qđ": "quyết định",
    "ct": "chỉ thị",
    "hđ": "hợp đồng",
    "tn": "thu nhập",
    "bh": "bảo hiểm",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "hcsn": "hành chính sự nghiệp",
    "tncn": "thu nhập cá nhân",
    "gtgt": "giá trị gia tăng",
    "tdn": "thu nhập doanh nghiệp",
    "tnc": "thu nhập chịu thuế",
    "qt": "quyết toán",
}


def normalize_vietnamese(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_vietnamese_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS]


def expand_legal_abbreviations(text: str) -> str:
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_TERM_MAP, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    def replacer(m):
        return LEGAL_TERM_MAP[m.group(1).lower()]
    return pattern.sub(replacer, text)


def prepare_for_bm25(text: str) -> str:
    text = normalize_vietnamese(text)
    text = expand_legal_abbreviations(text)
    return text


def tokenize_legal(text: str, use_vn_segmentation: bool = True) -> list[str]:
    text = prepare_for_bm25(text)
    if use_vn_segmentation:
        try:
            from src.retrieval.segmentation import tokenize_vietnamese
            tokens = tokenize_vietnamese(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()
    tokens = remove_stopwords(tokens)
    return tokens


DOC_CODE_PATTERNS = [
    re.compile(r"(\d+/\d{4}/QH\d+)"),
    re.compile(r"(\d+/\d{4}/NĐ-\w+)"),
    re.compile(r"(\d+/\d{4}/TT-\w+)"),
    re.compile(r"(\d+/\d{4}/CTN)"),
    re.compile(r"(\d+/\d{4}/NQ-\w+)"),
    re.compile(r"(\d+/\d{4}/QĐ-\w+)"),
    re.compile(r"(\d+/\d{4}/CT-\w+)"),
    re.compile(r"(\d+/NQ-\w+)"),
    re.compile(r"(\d+/QĐ-\w+)"),
    re.compile(r"(\d+/CT-\w+)"),
    re.compile(r"(\d+/NĐ-\w+)"),
    re.compile(r"(\d+/TT-\w+)"),
]


def extract_doc_code(title: str) -> str:
    if not title:
        return ""
    for pat in DOC_CODE_PATTERNS:
        m = pat.search(title)
        if m:
            return m.group(1)
    return ""


def extract_article_references(text: str) -> list[dict]:
    patterns = [
        (r"Điều\s+(\d+)", "article"),
        (r"Khoản\s+(\d+)", "clause"),
        (r"Nghị định\s+(\d+/\d+/\w+(?:-\w+)?)", "decree"),
        (r"Thông tư\s+(\d+/\d+/\w+(?:-\w+)?)", "circular"),
        # dừng ở dấu câu/markdown/xuống dòng, không nuốt ";", "*", "."
        (r"Luật\s+([^,;.*\s]+(?:[ \t]+[^,;.*\s]+){0,3})", "law"),
        (r"Bộ luật\s+([^,;.*\s]+(?:[ \t]+[^,;.*\s]+){0,3})", "code"),
    ]
    refs = []
    for pattern, ref_type in patterns:
        for m in re.finditer(pattern, text):
            refs.append({"type": ref_type, "value": m.group(1).strip(), "match": m.group(0)})
    return refs


def extract_structured_references(text: str) -> list[dict]:
    combined_pattern = re.compile(r"Điều\s+(\d+)\s+Khoản\s+(\d+)")
    article_only_pattern = re.compile(r"Điều\s+(\d+)")
    clause_pattern = re.compile(r"Khoản\s+(\d+)")

    refs = []
    seen = set()
    matched_spans = []

    for m in combined_pattern.finditer(text):
        article_num = m.group(1)
        clause_num = m.group(2)
        key = f"{article_num}_{clause_num}"
        if key not in seen:
            seen.add(key)
            matched_spans.append((m.start(), m.end()))
            refs.append({
                "type": "article_clause",
                "article_id": article_num,
                "clause_id": clause_num,
                "match": f"Điều {article_num} Khoản {clause_num}",
            })

    for m in article_only_pattern.finditer(text):
        if any(s <= m.start() < e for s, e in matched_spans):
            continue
        article_num = m.group(1)
        key = f"art_{article_num}"
        if key not in seen:
            seen.add(key)
            refs.append({
                "type": "article",
                "article_id": article_num,
                "clause_id": "",
                "match": f"Điều {article_num}",
            })

    last_article = ""
    for m in article_only_pattern.finditer(text):
        if any(s <= m.start() < e for s, e in matched_spans):
            continue
        last_article = m.group(1)

    for m in clause_pattern.finditer(text):
        if any(s <= m.start() < e for s, e in matched_spans):
            continue
        clause_num = m.group(1)
        before = text[max(0, m.start() - 20):m.start()]
        if not article_only_pattern.search(before) and last_article:
            key = f"{last_article}_{clause_num}"
            if key not in seen:
                seen.add(key)
                refs.append({
                    "type": "article_clause",
                    "article_id": last_article,
                    "clause_id": clause_num,
                    "match": f"Điều {last_article} Khoản {clause_num}",
                })

    clause_articles = {r["article_id"] for r in refs if r["type"] == "article_clause"}
    refs = [r for r in refs if not (r["type"] == "article" and r["article_id"] in clause_articles)]

    return refs
