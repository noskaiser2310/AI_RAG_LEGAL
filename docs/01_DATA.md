# DATA — Hệ thống dữ liệu pháp luật Việt Nam

> Mục tiêu: Xây dựng corpus pháp luật Việt Nam hoàn chỉnh cho bài toán Legal Retrieval & QA
> Nguồn: Public datasets (CC BY 4.0) + Crawl bổ sung

---

## 1. Tổng quan kiến trúc dữ liệu

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                                   │
│                                                                      │
│  HF Datasets                            Crawl                        │
│  ┌─────────────┐                  ┌──────────────┐                   │
│  │ Legal Docs  │                  │ VietLex API  │                   │
│  │ (153k docs) │                  │ (84k docs)   │                   │
│  ├─────────────┤                  ├──────────────┤                   │
│  │ Pháp Điển   │                  │ TVPL         │                   │
│  │ (202k arts) │                  │ (bổ sung)    │                   │
│  ├─────────────┤                  └──────────────┘                   │
│  │ UTS_VLC     │                                                    │
│  │ (318 laws)  │                    RAW CORPUS                       │
│  ├─────────────┤                 ┌──────────────┐                    │
│  │ PBGDPL Q&A  │                 │ ~200k VBPL   │                    │
│  │ (4,593 QA)  │                 │ ~300k Articles│                    │
│  └─────────────┘                 └──────┬───────┘                    │
│                                         │                            │
│                                  Cleaning & Normalization            │
│                                         │                            │
│                                         ▼                            │
│                               ┌──────────────────┐                   │
│                               │  Cleaned Corpus   │                   │
│                               │  (JSONL format)   │                   │
│                               └────────┬─────────┘                   │
│                                        │                             │
│                          ┌─────────────┼─────────────┐               │
│                          ▼             ▼             ▼               │
│                   ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│                   │ Chunk 1  │ │ Chunk 2  │ │ Chunk 3  │            │
│                   │ Article  │ │ Semantic │ │ Long     │            │
│                   │ Level    │ │ (512tok) │ │ (2048)   │            │
│                   └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│                        │            │            │                   │
│                        ▼            ▼            ▼                   │
│                   ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│                   │ FAISS    │ │ FAISS    │ │ BM25     │            │
│                   │ (Dense)  │ │ (Dense)  │ │ (Sparse) │            │
│                   └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết các nguồn dữ liệu

### 2.1. Nguồn chính (Primary Sources)

#### A. `th1nhng0/vietnamese-legal-documents` (⭐ Chính)
| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | vbpl.vn (Cổng QG về VBPL, Bộ Tư Pháp) |
| Số docs | 153,420 metadata + 178,665 content + 897,890 relationships |
| Format | 3 configs: `metadata`, `content`, `relationships` |
| License | CC BY 4.0 |
| Cập nhật | 2026 |
| Link | https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents |

**Schema metadata:**
```json
{
  "id": "61c8d174-8c1a-420f-bf6b-d04c1df501e9",
  "so_hieu": "04/2017/QH14",
  "loai_van_ban": "Luật",
  "co_quan_ban_hanh": "Quốc hội",
  "linh_vuc_phap_luat": "Doanh nghiệp",
  "ngay_ban_hanh": "2017-11-22",
  "ngay_co_hieu_luc": "2018-01-01",
  "tinh_trang_hieu_luc": "Còn hiệu lực",
  "trich_yeu": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
  "nguoi_ky": "Nguyễn Thị Kim Ngân",
  "pham_vi_dieu_chinh": "Toàn quốc",
  "thong_tin_ap_dung": "Áp dụng với mọi DNNVV"
}
```

**Schema content:**
```json
{
  "id": "61c8d174-...",
  "content_html": "<html>...toàn văn HTML...</html>"
}
```

**Schema relationships:**
```json
{
  "id": "...",
  "doc_id": "61c8d174-...",
  "other_doc_id": "e3f2a1b4-...",
  "relation_type": "sua_doi_bo_sung",
  "description": "Sửa đổi Điều 4"
}
```

#### B. `tmquan/phapdien-moj-gov-vn` (⭐ Article-level retrieval)
| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | phapdien.moj.gov.vn (Bộ Pháp Điển) |
| Số articles | 202,000+ |
| Format | Parquet + CSV: `articles`, `ontology_topics`, `ontology_subjects`, `ontology_glossary` |
| License | other (government data) |
| Link | https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn |

**Schema articles:**
```json
{
  "doc_id": "04/2017/QH14",
  "article_id": "Điều 4",
  "title": "Điều 4. Tiêu chí xác định doanh nghiệp nhỏ và vừa",
  "content": "Doanh nghiệp nhỏ và vừa được xác định theo các tiêu chí...",
  "chapter": "Chương I. NHỮNG QUY ĐỊNH CHUNG",
  "section": null,
  "subject": "doanh-nghiep",
  "topic": "doanh-nghiep-nho-va-vua"
}
```

#### C. `undertheseanlp/UTS_VLC`
| Thuộc tính | Giá trị |
|-----------|---------|
| Số docs | 318 (3 splits: 2021=110, 2023=208, 2026=318) |
| Tổng kích thước | ~24M characters |
| Kiểu | Hiến pháp + Bộ luật + Luật (1945-2025) |
| License | MIT |
| Link | https://huggingface.co/datasets/undertheseanlp/UTS_VLC |

```python
# Load
from datasets import load_dataset
ds = load_dataset("undertheseanlp/UTS_VLC", split="2026")

# Filter by type
codes = ds.filter(lambda x: x["type"] == "code")    # 6 Bộ luật
laws = ds.filter(lambda x: x["type"] == "law")      # 312 Luật
```

### 2.2. Nguồn bổ trợ (Auxiliary Sources)

#### D. `tmquan/pbgdpl-vn-legal-qna` (Validation data)
| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | pbgdpl.gov.vn (Bộ Tư Pháp) |
| Số Q&A | 4,593 cặp |
| Phân loại | 532 LinhVuc topics, 29 active |
| Trích dẫn | 98.1% câu trả lời viện dẫn ít nhất 1 VB |
| Link | https://huggingface.co/datasets/tmquan/pbgdpl-vn-legal-qna |

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?",
  "answer": "Doanh nghiệp được hỗ trợ khi...",
  "law_citations": ["04/2017/QH14", "80/2021/NĐ-CP"],
  "linh_vuc": "doanh-nghiep",
  "ngay_gui": "2021-01-15",
  "source_url": "https://pbgdpl.gov.vn/..."
}
```

#### E. `duyet/vietnamese-legal-instruct` (Fine-tuning data)
| Thuộc tính | Giá trị |
|-----------|---------|
| Số pairs | 467,732 training pairs |
| Kiểu | 14 QA types (summarize, classify, qa_practical, explain_simple, ...) |
| Format | CSV |
| Link | https://huggingface.co/datasets/duyet/vietnamese-legal-instruct |

#### F. `tmquan/anle-toaan-gov-vn` (Án lệ - optional)
| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | anle.toaan.gov.vn (TAND Tối cao) |
| Nội dung | Bản án + Án lệ |
| Format | Hierarchical: document → section → paragraph → sentence |
| Link | https://huggingface.co/datasets/tmquan/anle-toaan-gov-vn |

---

## 3. Data Quality Assessment

### 3.1. Chất lượng từng nguồn

| Source | Coverage | Completeness | Consistency | Freshness | Risk Level |
|--------|----------|-------------|-------------|-----------|------------|
| **A. th1nhng0/legal-docs** | ✅ 153k docs | ⚠️ Chỉ 178k content / 153k meta (một số VB thiếu nội dung) | ⚠️ UUID ID không khớp doc_id | ⚠️ Không rõ ngày cập nhật cuối | Medium |
| **B. phapdien-moj-gov-vn** | ✅ 202k articles | ✅ Article-level đầy đủ | ✅ Có doc_id, article_id chuẩn | ✅ Cập nhật từ phapdien.moj.gov.vn | Low |
| **C. UTS_VLC** | ⚠️ Chỉ 318 laws | ✅ Toàn văn | ✅ Curated bởi researcher | ✅ 3 splits (2021/2023/2026) | Low |
| **D. PBGDPL Q&A** | ⚠️ 29/532 topics active | ✅ 4,593 QA pairs | ⚠️ Cần verify answer quality | ✅ Dữ liệu mới | Medium |
| **E. vietnamese-legal-instruct** | ✅ 467k pairs | ✅ Đa dạng 14 QA types | ⚠️ Synthetic, cần filter type | - | Medium |
| **F. anle-toaan-gov-vn** | ⚠️ Không phải VBPL | N/A | N/A | N/A | Low priority |

### 3.2. Vấn đề chất lượng chi tiết

#### A. Overlap & Dedup giữa các nguồn

```python
# Các nguồn overlap đáng kể
# Strategy: merge by doc_id, phapdien làm primary (article-level sẵn)

overlap_matrix = {
    "A ∩ B": "~80% docs overlap (doc_id matching needed)",
    "A ∩ C": "~100% (UTS_VLC subset của A)",
    "B ∩ C": "~90% (phapdien có article-level, UTS_VLC có toàn văn)",
}
# Quy tắc ưu tiên: B (phapdien) > A (th1nhng0) > C (UTS_VLC)
```

**Vấn đề**: 
- A dùng UUID (vd: `61c8d174-...`), B dùng doc_id (vd: `04/2017/QH14`)  
- Cần mapping UUID ↔ doc_id qua trường `so_hieu` trong A metadata
- Articles trùng giữa A (extracted từ HTML) và B (structured) → giữ B

#### B. Document Validity Graph

```python
# VBPL thay đổi liên tục qua các VB sửa đổi
# Cần xây dựng directed graph để biết article nào còn hiệu lực

class ValidityGraph:
    """
    A sửa_đổi B → B thay thế Điều X của C
    
    Ví dụ: 
      Luật 04/2017/QH14  (gốc)
        ← Luật 03/2022/QH15 sửa_đổi Điều 4, Điều 7
        ← NĐ 80/2021/NĐ-CP hướng_dẫn Điều 10, Điều 15
        → [current state: Điều 4 mới, Điều 7 mới, còn lại giữ nguyên]
    
    Usage: khi question hỏi về "Điều 4 Luật Hỗ trợ DNNVV"
      → resolve đến phiên bản mới nhất (03/2022/QH15 sửa đổi)
      → Cần cả old + new version cho câu hỏi về timeline
    """
    def resolve_article(self, doc_id: str, article_id: str, as_of_date: str = None):
        """Trả về nội dung article tại thời điểm as_of_date"""
```

**Problems**:
- 897k relationships từ dataset A, cần parse để xây graph
- `tinh_trang_hieu_luc` trong A metadata có thể stale
- NĐ hướng_dẫn Luật → cần include cả NĐ khi Luật được viện dẫn

#### C. Article Extraction từ HTML (th1nhng0)

```python
# HTML từ vbpl.vn có cấu trúc không đồng nhất
# Một số vbpl dùng <p class="dieu">, một số dùng <div class="article">

html_variants = {
    "modern": '<div class="vb-content"><p class="dieu">Điều 4.</p><p>Nội dung...</p></div>',
    "legacy": '<table><tr><td>Điều 4: </td><td>Nội dung...</td></tr></table>',
    "raw": '<p><b>Điều 4.</b> Nội dung...</p>',
}

# Risk: extraction không chính xác → chunk sai → retrieval fail
# Giải pháp: dùng phapdien (article-level) làm ground truth,
# chỉ fallback sang HTML extraction khi phapdien không có
```

#### D. Article ID Normalization

| Variant | Chuẩn hóa về | Ghi chú |
|---------|-------------|---------|
| `Điều 4.` | `Điều 4` | Bỏ dấu chấm |
| `Điều IV` | `Điều 4` | La Mã → số |
| `điều 4` | `Điều 4` | Hoa đầu |
| `Điều thứ tư` | `Điều 4` | Chữ → số |
| `Điều 4.1` | `Khoản 1 Điều 4` | Chuẩn hóa về Khoản |
| `4.` (trong list) | `Khoản 4` | Nếu nằm trong article |

```python
def normalize_article_id(raw: str) -> str:
    """Chuẩn hóa article ID về format 'Điều X'"""
    roman_map = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,
                 'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}
    text_map = {'một':1,'hai':2,'ba':3,'bốn':4,'năm':5,'sáu':6,
                'bảy':7,'tám':8,'chín':9,'mười':10}
    # Rule-based normalization
    return f"Điều {number}"
```

#### E. PBGDPL Q&A Quality (Validation Set)

**Metrics cần kiểm tra trước khi dùng làm validation**:

| Check | Method | Expected | Risk |
|-------|--------|----------|------|
| Answer có hallucination không | LLM-as-a-Judge trên 100 mẫu | <5% hallucination | Nếu >10%, cần clean |
| Citation có đúng article không | Cross-check với corpus | >90% match | Nếu thấp, chỉ dùng doc-level |
| Topic coverage | Phân bố 29 topics | Đồng đều | Có thể bias 1-2 topics |
| Câu hỏi mập mờ | Human review 50 mẫu | <10% ambiguous | Cần remove hoặc clarify |
| Thời gian | `ngay_gui` từ 2021-2025 | Luật còn hiệu lực | Nếu citation dẫn luật cũ, cần update |

#### F. Duyet/vietnamese-legal-instruct Quality Filter

```python
# 467k pairs, 14 QA types — không phải tất cả đều useful

useful_types = {
    "qa_practical": "✅ Câu hỏi tình huống thực tế",
    "qa_legal": "✅ Hỏi đáp pháp luật",
    "explain_simple": "✅ Giải thích đơn giản",
    "qa_conditional": "✅ If-then scenario",
}
skip_types = {
    "summarize": "❌ Tóm tắt, không phải QA",
    "classify": "❌ Phân loại chủ đề",
    "extract_entities": "❌ Trích xuất thực thể",
}

# Risk: dùng wrong type → LLM học sai pattern
# Strategy: chỉ dùng 4/14 types cho fine-tuning
```

### 3.3. Chiến lược xử lý

#### Priority Document Sources

```
Source Priority (1 = highest):
  1. phapdien-moj-gov-vn (article-level, structured)
  2. th1nhng0/legal-docs (content fallback)
  3. UTS_VLC (validation / cross-check)
```

**Quy tắc merge**:
- Nếu doc_id có trong phapdien → dùng phapdien articles
- Nếu doc_id chỉ có trong th1nhng0 → parse HTML, extract articles
- Nếu doc_id trong cả 3 → phapdien > th1nhng0 > UTS_VLC
- Án lệ (F) → optional, chỉ dùng nếu cần context về interpretation

#### Validity-Aware Indexing

```python
def build_corpus_with_validity(sources: List[Dataset]) -> Corpus:
    """
    1. Merge docs, resolve conflicts
    2. Build validity graph từ relationships
    3. Mark articles với:
       - status: "còn hiệu lực" | "đã sửa đổi" | "đã thay thế" | "đã bãi bỏ"
       - replaced_by: doc_id + article_id (nếu có)
       - original_version: link đến article gốc
    4. Index tất cả versions (cho phép hỏi về luật cũ)
    """
    corpus = Corpus()
    
    for doc in merge_dedup(sources):
        for article in doc.articles:
            validity = resolve_validity(article, relationships)
            corpus.add(ArticleVersion(
                doc_id=doc.doc_id,
                article_id=article.article_id,
                content=augment_article_with_title(article),
                validity=validity,
                as_of_date=doc.effective_date,
                source_priority=get_priority(doc.source),
            ))
    
    return corpus
```

#### Validation Pipeline

```python
# Trước khi dùng PBGDPL Q&A làm validation set, chạy quality checks:

def validate_qna_dataset(qna: Dataset) -> Tuple[Dataset, Report]:
    issues = []
    clean = []
    
    for item in qna:
        checks = {
            "has_answer": len(item.answer) > 50,
            "has_citations": len(item.law_citations) > 0,
            "citations_exist": all(cite_in_corpus(c) for c in item.law_citations),
            "answer_not_hallucinated": not detect_hallucination(item.question, item.answer),
            "topic_known": item.linh_vuc in active_topics,
        }
        
        if all(checks.values()):
            clean.append(item)
        else:
            issues.append({"id": item.id, "failed_checks": [
                k for k, v in checks.items() if not v]})
    
    return clean, Report(
        total=len(qna), clean=len(clean), issues=len(issues),
        breakdown={k: sum(1 for i in issues if k in i["failed_checks"])
                   for k in checks.keys()}
    )
```

### 3.4. Data Quality Scorecard

| Dimension | Current State | Target | Action |
|-----------|--------------|--------|--------|
| **Article coverage** | ~300k articles | 500k+ (all VBPL) | Bổ sung từ phapdien; crawl thêm từ vbpl.vn |
| **Article extraction accuracy** | Unknown (HTML parse) | >95% match với phapdien | Test trên 1k mẫu; fallback to phapdien |
| **Validity tracking** | Not implemented | Full validity graph | Parse 897k relationships; build DAG |
| **Article ID normalization** | Not implemented | 100% consistent | Rule-based + regex pipeline |
| **Validation set quality** | Unknown | <5% hallucination, >90% citation accuracy | LLM-as-a-Judge audit |
| **Cross-source dedup** | Not implemented | 100% dedup | Merge by doc_id + article_id |
| **Topic coverage** | 29 active topics | Balanced distribution | Stratified sampling for eval |
| **Freshness** | Unknown | Corpus refresh ≤ 7 days before deadline | Schedule cron job |

---

## 4. Hệ thống phân loại văn bản pháp luật Việt Nam

### 3.1. Thứ bậc hiệu lực (Hierarchy)

| Level | Loại VB | Cơ quan ban hành | Ví dụ |
|-------|---------|-----------------|-------|
| 1 | **Hiến pháp** | Quốc hội | Hiến pháp 2013 |
| 2 | **Bộ luật, Luật** | Quốc hội | Luật Doanh nghiệp 2024 |
| 3 | **Pháp lệnh, Nghị quyết** | UBTVQH | Pháp lệnh xử lý VPHC |
| 4 | **Lệnh, Quyết định** | Chủ tịch nước | Lệnh công bố Luật |
| 5 | **Nghị định** | Chính phủ | NĐ 80/2021/NĐ-CP |
| 6 | **Quyết định** | Thủ tướng | QĐ 23/2021/QĐ-TTg |
| 7 | **Thông tư** | Bộ trưởng | TT 05/2021/TT-BKHĐT |
| 8 | **Nghị quyết** | HĐND Tỉnh | NQ 12/2021/HĐND |

### 3.2. Cấu trúc nội dung văn bản

```
Văn bản
├── Phần (Part)             — Optional, cho VB lớn (VD: BLDS có 6 Phần)
│   ├── Chương (Chapter)    — Bắt buộc với VB có nhiều Điều
│   │   ├── Mục (Section)  — Optional, nhóm các Điều cùng chủ đề
│   │   │   ├── Điều (Article) — Đơn vị cơ bản
│   │   │   │   ├── Khoản (Clause) — 1., 2., 3. ...
│   │   │   │   │   ├── Điểm (Point) — a), b), c) ...
│   │   │   │   │   │   └── Đoạn (Paragraph) — inline
```

### 3.3. Các mối quan hệ liên văn bản

| Relation | Ý nghĩa | Pattern |
|----------|---------|---------|
| `sua_doi` | Sửa đổi điều X | "sửa đổi Điều X" |
| `bo_sung` | Bổ sung khoản Y | "bổ sung Khoản Y" |
| `thay_the` | Thay thế điều Z | "thay thế Điều Z" |
| `can_cu` | Căn cứ điều W | "căn cứ Điều W" |
| `huong_dan` | Hướng dẫn thi hành | "hướng dẫn thi hành Điều V" |
| `bo_ban_hanh` | Bãi bỏ điều U | "bãi bỏ Điều U" |

---

## 5. Pipeline xử lý dữ liệu

### 5.1. Cleaning & Normalization

```python
def clean_legal_text(text: str) -> str:
    steps = [
        ("strip_html", strip_html_tags),
        ("normalize_unicode", normalize_vi_unicode),       # NFC normalization
        ("fix_typos", fix_vietnamese_typos),                # Sửa lỗi chính tả
        ("normalize_quotes", normalize_quotes),              # " " → ""
        ("remove_extra_whitespace", remove_extra_ws),
        ("normalize_citation", normalize_citations),         # Chuẩn hóa "điều 4" → "Điều 4"
        ("split_articles", split_into_articles),             # Tách articles
        ("extract_hierarchy", extract_doc_hierarchy),        # Phần → Chương → Điều
    ]
    for name, func in steps:
        text = func(text)
    return text
```

### 5.2. Hierarchical Chunking Strategy

```python
def hierarchical_chunk(article: Article, strategy: str) -> List[Chunk]:
    if strategy == "article_level":
        # Mỗi article = 1 chunk + title augmentation
        chunk = Chunk(
            content=f"[{article.doc_title}] - {article.chapter} - {article.article_id}: {article.content}",
            metadata={
                "doc_id": article.doc_id,
                "article_id": article.article_id,
                "chapter": article.chapter,
                "hierarchy_path": f"{article.doc_id}/{article.chapter}/{article.article_id}"
            }
        )
        return [chunk]
    
    elif strategy == "semantic":
        # Semantic splitting theo câu
        sentences = split_sentences(article.content)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(tokenize(sent))
            if current_len + sent_len > 512:
                chunks.append(merge_chunk(current_chunk, article))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len
        
        return chunks
    
    elif strategy == "long":
        # Overlap chunks cho articles dài > 2048 tokens
        return sliding_window(article.content, window=2048, stride=1792)
```

### 5.3. Chiến lược chunking tổng hợp

| Type | Size | Overlap | Purpose | Dùng cho |
|------|------|---------|---------|----------|
| **Article-level** | Full article | 0 | Exact article retrieval | Reranker, LLM context |
| **Semantic-granular** | 512 tokens | 128 | Query-to-phrase matching | Dense retrieval |
| **Long-granular** | 2048 tokens | 256 | Deep context | BM25, fallback |

### 5.4. Title Augmentation

```python
def augment_article_with_title(article: Article) -> str:
    """Augment article content with hierarchical titles"""
    title_chain = " | ".join([
        article.doc_title,           # "Luật Hỗ trợ DNNVV"
        article.chapter,             # "Chương I. NHỮNG QUY ĐỊNH CHUNG"
        article.section or "",       # optional
        article.article_id           # "Điều 4"
    ])
    return f"{title_chain}\n{article.content}"
```

---

## 6. Schema dữ liệu đầu ra

### 6.1. Processed Corpus (JSONL)

```json
{
  "chunk_id": "61c8d174--Điều_4--chunk_0",
  "doc_id": "04/2017/QH14",
  "article_id": "Điều 4",
  "doc_type": "Luật",
  "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
  "issuer": "Quốc hội",
  "issue_date": "2017-11-22",
  "effective_date": "2018-01-01",
  "status": "Còn hiệu lực",
  "chapter": "Chương I. NHỮNG QUY ĐỊNH CHUNG",
  "section": null,
  "content": "[Luật Hỗ trợ DNNVV] - Chương I - Điều 4: \n\n1. Doanh nghiệp nhỏ và vừa bao gồm...",
  "content_original": "Điều 4. Tiêu chí xác định doanh nghiệp nhỏ và vừa...",
  "hierarchy_path": "04/2017/QH14/Chương_I/Điều_4",
  "embedding_1024": [0.123, -0.456, ...],
  "tokens_count": 342,
  "source": "th1nhng0/vietnamese-legal-documents"
}
```

### 6.2. FAISS Index Schema (Multi-Embedding)

| Embedding Model | Dim | Index Type | Total Vectors | Size |
|----------------|-----|------------|---------------|------|
| vietlegal-harrier-0.6b | 1024 | `IndexIDMap(IndexFlatIP)` | ~300k | ~1.2 GB |
| vietlegal-e5 | 1024 | `IndexIDMap(IndexFlatIP)` | ~300k | ~1.2 GB |
| cotu-legal-retriever-Qwen3-Embedding-4B | 2560 | `IndexIDMap(IndexFlatIP)` | ~300k | ~3.0 GB |

```python
# Mỗi embedding model có index riêng
class MultiEmbeddingIndex:
    def __init__(self):
        self.indices = {
            "harrier": FAISSIndex(dim=1024, model="mainguyen9/vietlegal-harrier-0.6b"),
            "e5": FAISSIndex(dim=1024, model="mainguyen9/vietlegal-e5"),
            "qwen3_legal": FAISSIndex(dim=2560, model="minhnguyent546/cotu-legal-retriever-Qwen3-Embedding-4B-stage1"),
        }

    def search_all(self, query: str, k: int = 500) -> Dict[str, List[ScoredChunk]]:
        return {name: idx.search(query, k) for name, idx in self.indices.items()}
```

### 6.3. BM25 Index Schema

| Field | Type | Description |
|-------|------|-------------|
| tokenizer | `Underthesea + phobert` | Vietnamese word segmentation |
| bm25_params | `k1=1.5, b=0.75` | Okapi BM25 |
| corpus | `List[str]` | Article-level + semantic chunks |
| metadata | `List[Dict]` | Parallel metadata array |

---

## 7. Statistics & Coverage

### 7.1. Target Legal Domains (cho SME)

| Domain | Số articles mục tiêu | Luật chính |
|--------|---------------------|------------|
| **Doanh nghiệp** | 500+ | Luật Doanh nghiệp 2024, Luật Đầu tư |
| **Lao động** | 400+ | BLLĐ 2019, Luật BHXH, Luật Việc làm |
| **Thuế** | 600+ | Luật QL Thuế, Luật TNDN, GTGT, TNCN |
| **Hợp đồng** | 300+ | BLDS 2015 (Phần Hợp đồng) |
| **Đất đai** | 200+ | Luật Đất đai 2024 |
| **SHTT** | 150+ | Luật SHTT |

### 7.2. Dung lượng ước tính

| Component | Size | Notes |
|-----------|------|-------|
| Raw documents | ~20 GB | HTML + JSON |
| Cleaned corpus (JSONL) | ~5 GB | Processed text |
| FAISS index (harrier, 1024d) | ~1.2 GB | ~300k vectors |
| FAISS index (e5, 1024d) | ~1.2 GB | ~300k vectors |
| FAISS index (qwen3-legal, 2560d) | ~3.0 GB | ~300k vectors |
| BM25 index | ~3 GB | Sparse |
| Metadata store | ~500 MB | SQLite / JSON |
| **Total** | **~34 GB** | |

---

## 8. Cách load dữ liệu

```python
# 1. Load từ HuggingFace
from datasets import load_dataset

legal_docs = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
phapdien = load_dataset("tmquan/phapdien-moj-gov-vn", split="data")
qna_data = load_dataset("tmquan/pbgdpl-vn-legal-qna", split="data")

# 2. Filter theo lĩnh vực SME
sme_laws = legal_docs.filter(lambda x: x["linh_vuc_phap_luat"] in [
    "Doanh nghiệp", "Lao động", "Thuế", "Hợp đồng", "Đầu tư"
])

# 3. Load content tương ứng
content_ds = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data")
doc_ids = set(sme_laws["id"])
sme_content = content_ds.filter(lambda x: x["id"] in doc_ids)

# 4. Parse HTML → Text
from bs4 import BeautifulSoup
for item in sme_content:
    soup = BeautifulSoup(item["content_html"], "html.parser")
    clean_text = extract_legal_text(soup)  # Custom parser
```

---

## 9. Lưu ý quan trọng

1. **Copyright**: VBPL là tài sản công cộng (Điều 15 khoản 2 Luật SHTT 2005)
2. **License datasets**: CC BY 4.0 — dẫn nguồn khi sử dụng
3. **Cập nhật**: VBPL thay đổi liên tục; nên refresh corpus gần ngày thi
4. **Hiệu lực**: Cần filter `tinh_trang_hieu_luc == "Còn hiệu lực"` cho câu hỏi về luật hiện hành
5. **Quality**: ~98% articles có title, cần augment title cho articles thiếu
