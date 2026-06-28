# Kien truc He thong Legal RAG — R2AI2026

## Muc luc
1. [Tong quan](#1-tong-quan)
2. [Kien truc tong the](#2-kien-truc-tong-the)
3. [Module chi tiet](#3-module-chi-tiet)
4. [Agentic RAG](#4-agentic-rag)
5. [Danh gia trien khai](#5-danh-gia-trien-khai)
6. [De xuat cai thien](#6-de-xuat-cai-thien)

---

## 1. Tong quan

### Bai toan
- **Cuoc thi:** R2AI2026 BUILD AI LEGAL ASSISTANT
- **Nhiem vu:** Tra loi 2000 cau hoi phap luat tieng Viet, tra ve `relevant_docs` va `relevant_articles`
- **Metric chinh:** F2-Macro (beta=2, recall trong so gap 4x precision) + MRR
- **Dac diem test set:**
  - 99.8% cau hoi KHONG de cap ten van ban luat
  - Do dai TB: 37.8 tu, 46.6% cau >= 40 tu
  - 38.4% cross-doc (>= 2 van ban), 515 cau multi-hop
  - Phan loai: Thu tuc 23.8%, Dieu kien 20.2%, Quyen/Nghia vu 13.7%, Tinh huong 8.3%

### Nguyen tac thiet ke
- **Recall-first:** Uu tien tim du dieu luat (F2-beta=2), chap nhan precision thap hon
- **Garbage In = Garbage Out:** Truy xuat sai => Tra loi sai. Retrieval la nut that
- **Agentic:** LLM tu suy luan luat nao lien quan, tu danh gia du/thieu, tu tim them

---

## 2. Kien truc tong the

```
                         CAU HOI PHAP LUAT
                              |
                              v
              +-------------------------------+
              |   QUERY UNDERSTANDING          |
              |  +-------------------------+   |
              |  | 1. Classify (8 loai)    |   |
              |  | 2. Normalize + Expand   |   |
              |  | 3. Decompose (multi-hop)|   |
              |  +-------------------------+   |
              +-------------------------------+
                              |
              +-------------------------------+
              |   RETRIEVAL (Hybrid)           |
              |  +-------------------------+   |
              |  | Dense (FAISS + E5/Harrier)|  |
              |  | Sparse (BM25Okapi)      |   |
              |  | HyDE (hypothetical doc) |   |
              |  | Query Expansion (LLM)   |   |
              |  | Multi-hop (cross-ref)   |   |
              |  | RRF Fusion (weighted)   |   |
              |  +-------------------------+   |
              +-------------------------------+
                              |
              +-------------------------------+
              |   RERANK                        |
              |  +-------------------------+   |
              |  | Stage 1: CrossEncoder   |   |
              |  | Stage 2: LLM Listwise   |   |
              |  +-------------------------+   |
              +-------------------------------+
                              |
              +-------------------------------+
              |   POST-PROCESSING               |
              |  +-------------------------+   |
              |  | Article Reconstruction  |   |
              |  | (merge chunks -> Dieu)  |   |
              |  +-------------------------+   |
              +-------------------------------+
                              |
              +-------------------------------+
              |   GENERATION                    |
              |  +-------------------------+   |
              |  | LLM Generate + Context  |   |
              |  | Negative Detection      |   |
              |  | Self-Correction (2x)    |   |
              |  +-------------------------+   |
              +-------------------------------+
                              |
                              v
                    TRA LOI + TRICH DAN
```

### Hai che do hoat dong

| Che do | Phuong thuc | Mo ta |
|--------|-------------|-------|
| **One-shot** | `pipeline.answer()` | Classify -> Expand -> Retrieve 1 vong -> Rerank -> Generate |
| **Agentic** | `pipeline.answer_agentic()` | LLM recommend luat -> Retrieve nhieu vong -> Assess gap -> Follow-up -> Generate |

---

## 3. Module chi tiet

### 3.1 Data Loading (`src/data/loading.py`)

| Nguon | HF Dataset | So luong | Loai |
|-------|-----------|----------|------|
| Phap Dien | `tmquan/phapdien-moj-gov-vn` | Dieu luat | Article-level |
| Thu Vien PL | `vohuutridung/vietnamese-legal-documents` | Van ban | Full document |
| UTS_VLC | `undertheseanlp/UTS_VLC` | Van ban | Full document |
| Legal Pretrain | `KienCute/legal-pretrain` | Van ban | Full document |
| PBGDPL Q&A | `tmquan/pbgdpl-vn-legal-qna` | Hoi dap | Q&A pairs |

**Luu y:** Module `chunking.py` (legal structure-aware chunking theo Dieu/Chuong/Muc) **chua duoc su dung** trong pipeline thuc te. Cac loader tra ve pre-chunked docs.

### 3.2 Query Understanding

#### Query Classification (`src/retrieval/adaptive.py`)
8 loai cau hoi voi adaptive config rieng:

| Loai | Mo ta | top_k | HyDE | Dense/Sparse |
|------|-------|-------|------|-------------|
| `yes_no` | Co/Khong | 200/20 | Khong | 0.85/0.15 |
| `factual` | Su kien/quy dinh | 500/50 | Co | 0.75/0.25 |
| `condition` | Dieu kien/tieu chi | 600/60 | Co | 0.75/0.25 |
| `procedure` | Thu tuc/quy trinh | 600/60 | Co | 0.80/0.20 |
| `scenario` | Tinh huong gia dinh | 1000/80 | Co | 0.70/0.30 |
| `multi_article` | Tong hop nhieu dieu | 1000/80 | Co | 0.70/0.30 |
| `interpretation` | Giai thich/dien giai | 800/70 | Co | 0.90/0.10 |
| `comparison` | So sanh | 1000/80 | Co | 0.70/0.30 |

#### Query Expansion (`src/retrieval/query_expansion.py`)
- LLM sinh 1-4 bien the cau hoi (tuy loai)
- Moi bien the dung tu dong nghia/cach dien dat khac

#### Query Decomposition (`src/pipeline/orchestrator.py`)
- Tach cau hoi phuc tap thanh cau hoi con doc lap
- Ap dung cho: `multi_article`, `comparison`, `interpretation`, `scenario`
- Prompt huong dan 4 buoc phan tich phap ly

#### Text Processing (`src/retrieval/text_processor.py`)
- `normalize_vietnamese_light()`: NFKC + whitespace, giu nguyen dau/ca (cho LLM/embedding)
- `normalize_vietnamese()`: NFKC + strip punctuation + lowercase (cho BM25)
- `expand_legal_abbreviations()`: 30+ viet tat phap luat (BH->bao hiem, ND->nghi dinh...)
- `extract_article_references()`: Trich xuat "Dieu X", "Nghi dinh X/Y" tu answer

### 3.3 Retrieval

#### Dense Index (`src/retrieval/indexing.py`)
- **Model:** `mainguyen9/vietlegal-harrier-0.6b` (1024d) hoac `gemini-embedding-2` (3072d, fallback)
- **Engine:** FAISS `IndexFlatIP` + `IndexIDMap2`
- **Prefix:** E5-instruct "Voi mot truy van ve luat Viet Nam..."
- **Save/Load:** FAISS binary format

#### Sparse Index (`src/retrieval/indexing.py`)
- **Model:** BM25Okapi (k1=1.5, b=0.75)
- **Tokenization:** `underthesea` word segmentation + Vietnamese stopwords
- **Save/Load:** Pickle (BM25 object + corpus + doc_ids)

#### RRF Fusion (`src/retrieval/indexing.py`)
```
RRF_score(d) = sum(weight_i / (k + rank_i + 1)) for each ranked list i
```
- Dense weight: 0.70-0.90 (tuy loai cau hoi)
- Sparse weight: 0.10-0.30
- k = 60 (RRF constant)

#### Adaptive-K (`src/retrieval/multi_strategy.py`)
- Tim "gap lon nhat" trong score distribution
- Cat tail tai vi tri gap => tranh lay nhieu chunk khong lien quan

#### HyDE (`src/retrieval/hyde.py`)
- LLM sinh "van ban phap luat gia dinh" tra loi cau hoi
- Embed hypothetical doc => dense search => them chunks lien quan

#### Multi-hop Retrieval (`src/pipeline/orchestrator.py`)
- Regex phat hien cross-references: "Dieu X cua Luat nay", "Nghi dinh X/Y Dieu Z"
- Them follow-up queries tu cross-refs trong top 20 chunks
- Score penalty 0.85 cho multi-hop chunks

### 3.4 Reranker (`src/reranker/cross_encoder.py`)

#### Stage 1: CrossEncoder
- **Model:** `AITeamVN/Vietnamese_Reranker`
- **Calibration:** Softmax + MinMax scaling
- **Truncation:** Tim last space truoc 2048 chars (tranh cat giua tu tieng Viet)

#### Stage 2: LLM Reranker (listwise)
- LLM chon chunks lien quan tu batch 15 chunks
- Binary score: 1.0 (selected) / 0.0 (not selected)

#### TwoStageReranker
- CrossEncoder truoc (lay top_k*2) => LLM Reranker loc (lay top_k)

### 3.5 Generation (`src/generator/generator.py`)

- **System prompt:** Tro ly phap ly AI, tra loi dua tren context, trich dan dieu luat
- **Context formatting:** Dedup theo doc_title|article_id, max 1500 chars/chunk
- **Negative detection:** Phat hien cau tra loi "chung chung" => retry voi prompt manh hon
- **Self-correction (2 rounds):**
  - Round 1: Kiem tra chinh xac, trich dan, thieu thong tin
  - Round 2: Kiem tra sai sot, ngon ngu, trich dan can thiet
- **Score threshold:** Chon chunks co score >= 0.7, fallback top 20 neu < 5 chunks dat

### 3.6 Evaluation (`src/evaluation/metrics.py`)

- **F2-Macro:** F-beta voi beta=2, macro average tren queries
- **Recall@k, Precision@k:** k = 5, 10, 20, 50
- **MRR:** Mean Reciprocal Rank

---

## 4. Agentic RAG

### Kien truc

```
CAU HOI
  |
  v
[Step 1] LAW ANALYSIS (LLM)
  | "Luat Lao dong 2019, Dieu 36, Nghi dinh 145..."
  v
[Step 2] INITIAL RETRIEVAL
  | Query goc + search_queries tu Step 1
  | Dense + BM25 + RRF Fusion
  v
[Step 3] ASSESS COVERAGE (LLM)  <--+
  | "is_complete? gaps? follow_up?"    |
  v                                    |
[Step 4] TARGETED FOLLOW-UP           |
  | LLM-generate queries cu the      |
  | Retrieve bo sung                  |
  v                                    |
[Step 5] RE-ASSESS (max 2 vong) -----+
  v
[Step 6] RERANK (CrossEncoder + LLM)
  v
[Step 7] ARTICLE RECONSTRUCTION
  v
[Step 8] GENERATE + SELF-CORRECT
  v
TRA LOI + TRICH DAN
```

### Prompts chinh

| Prompt | Muc dich | Input | Output |
|--------|---------|-------|--------|
| `LAW_ANALYSIS_PROMPT` | Recommend luat | Cau hoi | domains, laws, key_articles, search_queries |
| `ASSESS_PROMPT` | Danh gia du/thieu | Cau hoi + context | is_complete, confidence, gaps, follow_up_queries |

### Dong xuly (`src/pipeline/agent.py`)

```python
class LegalAgent:
    async def answer(query, max_iterations=2):
        # 1. LLM phan tich => goi y luat
        law_analysis = await self._analyze_laws(query)

        # 2. Retrieve ban dau (query goc + search_queries)
        all_chunks = await self._retrieve_with_queries(
            [query] + search_queries, seen_ids
        )

        # 3-5. Lap agentic (max 2 vong)
        for iteration in range(max_iterations):
            reranked = await reranker.rerank(query, all_chunks)
            assessment = await self._assess_coverage(query, reranked)
            if assessment["is_complete"]: break
            new_chunks = await self._retrieve_with_queries(
                assessment["follow_up_queries"], seen_ids
            )
            all_chunks.extend(new_chunks)

        # 6-8. Rerank + Reconstruct + Generate
        final = await reranker.rerank(query, all_chunks, top_k)
        final = await self._reconstruct_articles(final)
        answer = await generator.generate(query, final)
```

### Parallel retrieval
- `_retrieve_with_queries()` dung `asyncio.gather()` de embed+retrieve song song nhieu queries
- Dedup bang `seen_ids` set

---

## 5. Danh gia trien khai

### Trang thai module

| Module | File | Trang thai | Ghi chu |
|--------|------|-----------|---------|
| **Core base** | `src/core/base.py` | OK | 8 QueryType, data contracts, abstract classes |
| **Config** | `src/core/config.py` | OK | Pydantic settings, .env integration |
| **Data loading** | `src/data/loading.py` | OK (4/5 nguon) | Thieu `anle` (an le toa an) |
| **Chunking** | `src/retrieval/chunking.py` | CHUA DUNG | Legal structure-aware, khong duoc goi |
| **Dense index** | `src/retrieval/indexing.py` | OK | FAISS + save/load |
| **Sparse index** | `src/retrieval/indexing.py` | OK | BM25 + pickle save/load |
| **RRF Fusion** | `src/retrieval/indexing.py` | OK | Weighted RRF |
| **Multi-strategy** | `src/retrieval/multi_strategy.py` | OK | Adaptive-K + hybrid search |
| **Query expansion** | `src/retrieval/query_expansion.py` | OK | LLM-based |
| **HyDE** | `src/retrieval/hyde.py` | OK | Hypothetical document |
| **Classifier** | `src/retrieval/adaptive.py` | OK | 8 types + adaptive config |
| **Segmentation** | `src/retrieval/segmentation.py` | OK | LRU cache, underthesea |
| **Text processor** | `src/retrieval/text_processor.py` | OK | Light + full normalization |
| **CrossEncoder** | `src/reranker/cross_encoder.py` | OK | Softmax + minmax |
| **LLM Reranker** | `src/reranker/cross_encoder.py` | OK | Listwise selection |
| **TwoStage** | `src/reranker/cross_encoder.py` | OK | CE + LLM pipeline |
| **Generator** | `src/generator/generator.py` | OK | Self-correct + negative detect |
| **Orchestrator** | `src/pipeline/orchestrator.py` | OK | One-shot pipeline |
| **Agentic** | `src/pipeline/agent.py` | OK | Multi-round LLM-guided |
| **Metrics** | `src/evaluation/metrics.py` | OK | F2, Recall, Precision, MRR |
| **Submit** | `scripts/submit.py` | OK | Agentic + oneshot modes |
| **Tune** | `scripts/tune.py` | LOI | Import sai, ground truth trong |
| **Run pipeline** | `scripts/run_pipeline.py` | OK | Test 3 cau hoi |

### Bug da sua
- [x] `_prepare_query` tra ve `expanded` thay vi `query` goc
- [x] `GeminiClient()` -> `Gemma4Client()` (import fix)
- [x] Score threshold 0.95 -> 0.7 (dead code fix)
- [x] Citations type mismatch (dict -> str)
- [x] Unbounded dict cache -> LRU cache (50k)
- [x] CrossEncoder truncation cat giua tu -> tim last space
- [x] HarrierEmbedding lam default
- [x] TwoStageReranker wired vao `create()`
- [x] SparseIndex save/load BM25 object
- [x] Article reconstruction skip empty article_id
- [x] `normalize_vietnamese_light()` bao toan dau/ca

### Loi chua sua
- [ ] `tune.py` import `adaptive_k_by_gap` tu `indexing` thay vi `multi_strategy`
- [ ] `tune.py` ground truth la `[[] for _ in queries]` => F2 luon = 0
- [ ] `chunking.py` khong duoc su dung (wasted code)
- [ ] `.env` chua API key (security risk)
- [ ] Khong co test nao (`tests/` khong ton tai)

---

## 6. De xuat cai thien

### Uu tien cao (Impact lon den ket qua thi)

#### 6.1 Chunking theo cau truc Dieu (CRITICAL)
**Van de:** `chunking.py` viet san nhung khong duoc dung. Cac loader tra ve pre-chunked docs co the khong theo ranh gioi Dieu/Khoan/Điem.

**Giai phap:**
- Wire `chunking.py` vao `build_corpus()` trong `loading.py`
- Tach moi Dieu thanh chunks theo Khoan/Điem (moi khoan = 1 chunk)
- Dieu ngan (< 500 tu) giu nguyen, Dieu dai tach theo khoan
- Metadata: `chunk_type: "dieu"|"khoan"|"diem"`, `parent_article_id`

**Impact:** Cai thien retrieval accuracy vi moi chunk = 1 don vi phap ly hoan chinh.

#### 6.2 Metadata `so_ky_hieu` cho submission format (CRITICAL)
**Van de:** `submit.py` can `format_doc_id()` tra ve `"ma VB|ten VB"` nhung `RetrievedChunk.metadata` khong co `so_ky_hieu`. Cac loaders khong trich xuat field nay.

**Giai phap:**
- Them regex trich xuat so ky hieu tu title: `Luat 04/2017/QH14`, `Nghi dinh 80/2021/ND-CP`
- Populate `metadata["so_ky_hieu"]` trong moi loader
- Fallback: parse tu `doc_title` bang regex khi format submission

**Impact:** Khong co ma VB => `relevant_docs` va `relevant_articles` sai format => diem = 0.

#### 6.3 Fix tune.py (HIGH)
**Van de:**
- Import sai: `from src.retrieval.indexing import adaptive_k_by_gap` (thuc te o `multi_strategy`)
- Ground truth trong `[[] for _ in queries]` => F2 luon = 0 => khong chon duoc params toi uu

**Giai phap:**
- Sua import
- Dung mot tap nho (50-100 cau) co ground truth tu tap PBGDPL Q&A de tune

#### 6.4 Corpus chat luong cao (HIGH)
**Van de:** 5 nguon du lieu co the trung lap, chat luong khong deu. `vohuutridung` la full documents (khong chia theo Dieu).

**Giai phap:**
- Uu tien: phapdien (article-level) > vohuutridung (full docs, can re-chunk) > utslvc
- Dedup trung giua cac nguon (cung ten VB + cung Dieu)
- Chi giu ban hop nhat (consolidated) khi co nhieu phien ban
- Filter bo chunks < 30 tu (khong du thong tin)

#### 6.5 Law Name Index (HIGH)
**Van de:** LLM co the goi y ten luat khong co trong corpus. Agent tim khong thay => waste iterations.

**Giai phap:**
- Xay dung index tat ca ten VB + so ky hieu tu corpus
- Khi LLM recommend luat, fuzzy match voi index
- Chi retrieve nhung luat co that trong corpus
- Log warning neu LLM goi y luat khong ton tai

### Uu tien trung binh

#### 6.6 Embedding model tot hon
**Hien tai:** `vietlegal-harrier-0.6b` (0.6B params, 1024d)
**De xuat:** Thu `BGE-m3` (multi-lingual, 1024d) hoac fine-tune harrier tren tap Q&A phap luat
**Ly do:** BTC nhac BGE-m3 dat F2=0.4783 tren VLQA benchmark

#### 6.7 Reranker model tot hon
**Hien tai:** `AITeamVN/Vietnamese_Reranker`
**De xuat:** `Qwen3-Reranker-0.6B` hoac fine-tune cross-encoder tren phap luat VN
**Ly do:** BTC nhac VLQA three-stage voi LLM reranker dat F2=0.7283

#### 6.8 LLM manh hon
**Hien tai:** `gemma-4-26b-a4b-it` (4B active params)
**De xuat:** `Qwen3-30B-A3B` (3B active, MoE) hoac `GPT-4o-mini` (neu co API)
**Ly do:** BTC benchmark Qwen3 dat F2=0.7283 + ROUGE-L 0.66 tren VLQA

#### 6.9 Two-stage retrieval (tu doc BTC)
BTC nhac rang VLQA three-stage retriever (BM25 -> Dense -> LLM reranker) dat hieu qua cao nhat. Hien tai chung ta da co two-stage (CrossEncoder + LLM) nhung co the:
- Them stage 0: BM25-only pre-filter (lay top 1000)
- Stage 1: Dense retrieval tu BM25 candidates
- Stage 2: CrossEncoder rerank
- Stage 3: LLM rerank (chi tren top 30-50)

#### 6.10 Article-level metadata cho relevant_articles
**Van de:** `relevant_articles` can format `"ma VB|ten VB|Dieu X"`. Hien tai chi co `article_id` la so (vd: "36"), khong co day du ten VB + ma so ky hieu.

**Giai phap:**
- Khi build corpus, trich xuat: `doc_code` (vd: "04/2017/QH14"), `doc_full_name` (vd: "Luat Ho tro doanh nghiep nho va vua")
- Khi build submission: format `"04/2017/QH14|Luat Ho tro DNNVV|Dieu 5"`

### Uu tien thap

#### 6.11 Unit tests
- Test JSON parser, text processor, article reference extraction
- Test RRF fusion, adaptive-k
- Test query classifier voi 20 cau mau

#### 6.12 Streaming/API server
- FastAPI endpoint cho demo
- Gradio UI cho testing truc quan

#### 6.13 Dynamic corpus update
- Tu dong crawl vbpl.vn khi co VB moi
- Re-index incrementally

---

## Phu luc: So sanh One-shot vs Agentic

| Yeu to | One-shot | Agentic |
|--------|----------|---------|
| Law recommendation | Khong | Co (LLM goi y ten luat) |
| Retrieval rounds | 1 (static) | 2-3 (dynamic) |
| Gap assessment | Khong | Co (LLM danh gia du/thieu) |
| Follow-up queries | Regex cross-ref | LLM-generated, co context |
| LLM calls (min) | ~5 | ~7 |
| LLM calls (max) | ~15 | ~20 |
| Latency | ~10-30s | ~20-50s |
| Coverage (uoc tinh) | 60-70% | 80-90% |

## Phu luc: So sanh voi BTC Recommendations

| BTC Recommend | Hien tai | Gap |
|---------------|----------|-----|
| Hybrid retrieval (BM25 + Dense) | OK | -- |
| RRF Fusion | OK | -- |
| Cross-encoder reranker | OK | -- |
| LLM-as-reranker (3-stage) | OK (TwoStage) | Them stage 0 BM25 pre-filter |
| Fine-tuned models cho VN | Harrier (pretrained) | Chua fine-tune |
| Article-level chunking | Khong dung | Can wire chunking.py |
| Multi-hop retrieval | OK (regex + agentic) | -- |
| Query decomposition | OK | -- |
| F2-Macro optimization | OK (recall-first) | -- |
| Qwen3 / BGE-m3 | Gemma4 + Harrier | Can upgrade model |
