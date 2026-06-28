# Architecture Documentation — Legal RAG

> Hệ thống Legal RAG cho cuộc thi **R2AI2026 BUILD AI LEGAL ASSISTANT**
> Mục tiêu: **Top 1** — F2-Macro trên tập test BTCC

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Query       │  │  Query       │  │  Adaptive    │  │  Result      │   │
│  │  Classifier  │  │  Expander    │  │  Retriever   │  │  Assembler   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
└──────────┼────────────────┼─────────────────┼──────────────────────────────┘
           │                │                  │
           ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL LAYER                                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    Multi-Strategy Retriever                       │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │      │
│  │  │  Dense   │  │  BM25    │  │  HyDE    │  │  Query Expanded  │  │      │
│  │  │  Search  │  │  Search  │  │  Search  │  │  Search          │  │      │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │      │
│  │       └──────────────┴────────────┴──────────────────┘             │      │
│  │                            │                                        │      │
│  │                            ▼                                        │      │
│  │                    ┌──────────────┐                                 │      │
│  │                    │  RRF Fusion  │                                 │      │
│  │                    └──────────────┘                                 │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                       RERANKER LAYER                             │      │
│  │  ┌─────────────────────┐  ┌────────────────────────────────────┐ │      │
│  │  │ Cross-Encoder       │  │ LLM-based Re-ranker (optional)     │ │      │
│  │  │ (Qwen3-Reranker)    │  │ (Gemini/Qwen3)                     │ │      │
│  │  └─────────────────────┘  └────────────────────────────────────┘ │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GENERATION LAYER                                   │
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐  │
│  │  Context     │  │  LLM Generation      │  │  Self-Correction Loop    │  │
│  │  Formatting  │──▶  (Gemini / Qwen3)    │──▶  (2 rounds max)         │  │
│  └──────────────┘  └──────────────────────┘  └──────────┬───────────────┘  │
│                                                          │                  │
│                                                          ▼                  │
│                                               ┌──────────────────────┐     │
│                                               │  Citation Extraction │     │
│                                               └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINAL ANSWER                                      │
│   + confidence score, citations, relevant articles                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Core (`src/core/`)

| File | Class/Function | Responsibility |
|------|----------------|----------------|
| `base.py` | `Message`, `RetrievedChunk`, `QueryResult` | Data contracts |
| `base.py` | `BaseLLM`, `BaseEmbedding`, `BaseRetriever`, `BaseReranker` | Abstract base classes (Adapter pattern) |
| `base.py` | `BaseQueryExpander`, `BaseQueryClassifier` | Strategy interfaces |
| `base.py` | `QueryType` enum | 6 query types for adaptive retrieval |
| `config.py` | `Config` | Pydantic settings (env vars, model paths, hyperparams) |

**Adapter Pattern**: Mỗi component (LLM, Embedding, Retriever, Reranker) có một abstract base class. Triển khai cụ thể chỉ cần kế thừa và implement các method.

```
BaseLLM ──┬── GeminiClient
          └── Qwen3Client (future)

BaseEmbedding ──┬── HarrierEmbedding
                └── BGEM3Embedding (future)

BaseReranker ──┬── CrossEncoderReranker
               ├── LLMReranker
               └── TwoStageReranker
```

### 2.2 LLM (`src/llm/`)

| File | Class | Description |
|------|-------|-------------|
| `gemini_client.py` | `GeminiClient` | Adapter cho Google Gemini API (dùng cho prototyping) |

**Key design decisions:**
- `generate()` returns `LLMResponse` with token counts
- `generate_stream()` for streaming use cases
- System message extraction từ messages list
- Temperature, max_tokens configurable per-call

### 2.3 Embedding (`src/embedding/`)

| File | Class | Description |
|------|-------|-------------|
| `harrier_embedding.py` | `HarrierEmbedding` | `mainguyen9/vietlegal-harrier-0.6b` — Dense embedding SOTA cho Vietnamese legal |

**Specs:**
- Model: vietlegal-harrier-0.6b
- Dimension: 1024
- Max length: 8192 tokens
- Normalization: L2 (inner product search)
- Device: configurable (cpu/cuda)

### 2.4 Retrieval (`src/retrieval/`)

| File | Class/Function | Description |
|------|----------------|-------------|
| `chunking.py` | `chunk_text()`, `split_by_legal_structure()` | Structure-aware chunking: hierarchical by Điều/Chương, sliding window với sentence boundary |
| `indexing.py` | `DenseIndex`, `SparseIndex` | FAISS (IP) + BM25 index builders |
| `indexing.py` | `build_indexes()` | Async index construction |
| `indexing.py` | `rrf_fusion()` | Reciprocal Rank Fusion over N ranked lists |
| `multi_strategy.py` | `MultiStrategyRetriever` | Unified retriever: dense + sparse + HyDE + expanded queries |
| `query_expansion.py` | `LLMQueryExpander` | LLM-based query expansion (sinh N query variations) |
| `hyde.py` | `HyDEGenerator` | Hypothetical Document Embedding — sinh văn bản pháp luật giả định, encode, search |
| `adaptive.py` | `LLMQueryClassifier` | Phân loại query vào 6 types |
| `adaptive.py` | `get_adaptive_config()` | Lấy config động dựa trên query type |

**Multi-Strategy Retrieval Flow:**
```
Original Query
    │
    ├──▶ Dense Search (Harrier embedding ▶ FAISS IP search)
    │
    ├──▶ BM25 Search (tokenized ▶ BM25Okapi)
    │
    ├──▶ [Query Expansion] ──▶ N variations ──▶ Each: Dense + BM25
    │
    └──▶ [HyDE] ──▶ LLM sinh văn bản mẫu ──▶ Harrier encode ──▶ Dense Search
         │
         ▼
    RRF Fusion ──▶ Merged ranked list
```

**Adaptive Query Types:**

| Type | Description | top_k_retrieval | num_variations | use_hyde |
|------|-------------|-----------------|----------------|----------|
| `yes_no` | Câu hỏi có/không | 200 | 1 | No |
| `factual` | Sự kiện, quy định cụ thể | 500 | 2 | Yes |
| `multi_article` | Tổng hợp nhiều điều luật | 1000 | 4 | Yes |
| `interpretation` | Giải thích, diễn giải | 800 | 3 | Yes |
| `procedure` | Thủ tục, quy trình | 600 | 2 | Yes |
| `comparison` | So sánh quy định | 1000 | 4 | Yes |

### 2.5 Reranker (`src/reranker/`)

| File | Class | Description |
|------|-------|-------------|
| `cross_encoder.py` | `CrossEncoderReranker` | Cross-encoder với score calibration (minmax scaling) |
| `cross_encoder.py` | `LLMReranker` | LLM-based pointwise reranking (0-10 scale) |
| `cross_encoder.py` | `TwoStageReranker` | Stage 1: Cross-encoder → Stage 2: LLM reranker |

**Điểm mạnh:**
- Softmax calibration: chuyển logits thành probability scores
- MinMax scaling: chuẩn hóa scores về [0,1] để dễ threshold
- Two-stage: lightweight model filter trước, LLM fine-rank sau

### 2.6 Generator (`src/generator/`)

| File | Class | Description |
|------|-------|-------------|
| `generator.py` | `Generator` | Context formatting + LLM generation + self-correction loop |

**Self-Correction Loop:**
```
Round 1: Kiểm tra tính chính xác, trích dẫn, đầy đủ
Round 2: Rà soát lần cuối (chỉ nếu round 1 có sửa)
         │
         ├── LLM trả lời "OK" → chấp nhận
         └── LLM trả lời khác → dùng câu trả lời đã sửa
```

**Citation Extraction:**
Regex patterns để trích xuất: Điều, Nghị định, Thông tư, Luật, Bộ luật từ câu trả lời.

### 2.7 Pipeline (`src/pipeline/`)

| File | Class | Description |
|------|-------|-------------|
| `orchestrator.py` | `LegalRAGPipeline` | Async orchestrator: kết nối tất cả components |

**Lifecycle:**
```
pipeline = await LegalRAGPipeline.create(docs, dense_path="...", sparse_path="...")
result = await pipeline.answer("Người lao động có được hưởng lương thử việc?")
# result.query_type, result.final_answer, result.citations, result.confidence
```

### 2.8 Evaluation (`src/evaluation/`)

| File | Function | Description |
|------|----------|-------------|
| `metrics.py` | `compute_f2_macro()` | F2-Macro calculation (β=2, recall 2x precision) |
| `metrics.py` | `compute_retrieval_metrics()` | Recall@k, Precision@k |
| `metrics.py` | `evaluate_answers()` | Full evaluation suite |

**F2-Macro**: Macro-average F2 score. Mỗi query được tính F2 riêng, sau đó average. β=2 có nghĩa recall được coi trọng gấp đôi precision.

### 2.9 Data (`src/data/`)

| File | Function | Description |
|------|----------|-------------|
| `loading.py` | `build_corpus()` | Load từ HuggingFace datasets → chunk → save JSONL |
| `loading.py` | `load_corpus()` | Load cached corpus từ JSONL |

---

## 3. Data Flow (per query)

```
Input: "Người lao động có được hưởng lương thử việc không?"
  │
  ├─ 1. Classify Query ──▶ type="yes_no" (hoặc "factual")
  │
  ├─ 2. Expand Query ──▶ variations: [
  │      "Quy định về tiền lương thử việc?",
  │      "Người lao động thử việc có quyền lợi gì?"
  │    ]
  │
  ├─ 3. Embed ──▶ Harrier ▶ [vec1(1024d), vec2(1024d), vec3(1024d)]
  │
  ├─ 4. Multi-Strategy Retrieval:
  │      ├── Dense: FAISS IP search (top-500)
  │      ├── BM25: tokenized search (top-500)
  │      ├── Expanded: dense+sparse cho mỗi variation
  │      └── HyDE: sinh văn bản mẫu → embed → dense search (top-250)
  │      └── RRF Fusion (k=60) → merged top-500
  │
  ├─ 5. Rerank ──▶ Qwen3-Reranker-0.6B (top-50)
  │
  ├─ 6. Score Filter ──▶ giữ chunks có score > 0.95 (hoặc top-20)
  │
  ├─ 7. Format Context ──▶ [1] Bộ luật Lao động - Điều 25: ...
  │                        [2] Bộ luật Lao động - Điều 26: ...
  │
  ├─ 8. Generate ──▶ Gemini/Qwen3 với system prompt + context
  │
  ├─ 9. Self-Correct ──▶ Round 1: kiểm tra → OK? hoặc sửa
  │                      Round 2: rà soát lần cuối (nếu cần)
  │
  ├─ 10. Extract Citations ──▶ ["Điều 25", "Điều 26", "Bộ luật Lao động"]
  │
Output: {
  answer: "Có, theo Điều 25 Bộ luật Lao động...",
  confidence: 0.97,
  citations: ["Điều 25", "Điều 26"],
  relevant_articles: ["25", "26"],
  query_type: "yes_no",
  correction_rounds: 1,
  retrieval_time: 2.3s,
  rerank_time: 0.8s,
  generation_time: 1.5s
}
```

---

## 4. Models & Configuration

### Current Models (Prototyping)

| Component | Model | Source |
|-----------|-------|--------|
| LLM | `gemini-2.0-flash` (prototyping) | Google API |
| Embedding | `mainguyen9/vietlegal-harrier-0.6b` | HuggingFace |
| Reranker | `AITeamVN/Vietnamese_Reranker` (tạm thời) | HuggingFace |

### Target Models (Final)

| Component | Model | VRAM | Notes |
|-----------|-------|------|-------|
| LLM | `Qwen3-8B-Instruct` (4-bit) | ~6 GB | Primary LLM |
| LLM (backup) | `thangvip/qwen3-4b-vietnamese-legal-grpo` | ~4 GB | Legal-tuned backup |
| Embedding | `mainguyen9/vietlegal-harrier-0.6b` | ~2 GB | NDCG@10=0.7813 |
| Reranker | `Qwen3-Reranker-0.6B` | ~1.5 GB | MTEB-R=65.80 |

### Key Config Parameters (`src/core/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_DIM` | 1024 | Harrier output dimension |
| `BM25_K1` | 1.5 | BM25 term saturation |
| `BM25_B` | 0.75 | BM25 length normalization |
| `RETRIEVAL_TOP_K` | 500 | Initial retrieval depth |
| `RERANK_TOP_K` | 50 | After reranker |
| `FINAL_TOP_K` | 20 | Final context window |
| `SCORE_THRESHOLD` | 0.95 | Relevance score cutoff |
| `TEMPERATURE` | 0.1 | LLM generation temperature |

---

## 5. Project Structure

```
D:\AI_RAG_LEGAL\
├── src/
│   ├── __init__.py
│   ├── core/           # Base classes, config, data contracts
│   │   ├── base.py
│   │   └── config.py
│   ├── data/           # Data loading, preprocessing, chunking
│   │   └── loading.py
│   ├── embedding/      # Embedding model adapters
│   │   └── harrier_embedding.py
│   ├── llm/            # LLM adapters (Gemini, Qwen3)
│   │   └── gemini_client.py
│   ├── retrieval/      # Multi-strategy retrieval
│   │   ├── adaptive.py       # Query classification + adaptive config
│   │   ├── chunking.py       # Legal structure-aware chunking
│   │   ├── hyde.py           # Hypothetical Document Embedding
│   │   ├── indexing.py       # FAISS + BM25 index builders
│   │   ├── multi_strategy.py # Unified multi-strategy retriever
│   │   └── query_expansion.py# LLM-based query expansion
│   ├── reranker/       # Cross-encoder + LLM rerankers
│   │   └── cross_encoder.py
│   ├── generator/      # LLM generation + self-correction
│   │   └── generator.py
│   ├── pipeline/       # Orchestrator
│   │   └── orchestrator.py
│   └── evaluation/     # F2-Macro metrics
│       └── metrics.py
├── data/
│   ├── raw/            # Raw downloaded data
│   ├── processed/      # Chunked JSONL corpus
│   ├── indexes/        # FAISS + BM25 persisted indexes
│   └── results/        # Evaluation outputs
├── scripts/
│   └── run_pipeline.py # Entry point
├── tests/              # Unit tests
├── docs/               # Documentation & research
│   ├── 01_DATA.md
│   ├── 02_RETRIEVAL_MODEL.md
│   ├── 03_RERANKER_MODEL.md
│   └── 04_GENERATION_MODEL.md
├── research/           # Cloned open-source repos
│   ├── ViDrill/
│   ├── URAxLaws/
│   └── ... (12 repos)
├── pyproject.toml
└── R2AI2026_LEGAL_ASSISTANT.md
```

---

## 6. Extension Guide

### Thêm LLM mới

```python
# src/llm/qwen3_client.py
from src.core.base import BaseLLM, LLMResponse, Message

class Qwen3Client(BaseLLM):
    async def generate(self, messages, **kwargs) -> LLMResponse:
        # Implement Qwen3 8B inference
        ...

    async def generate_stream(self, messages, **kwargs) -> str:
        ...
```

### Thêm Embedding mới

```python
from src.core.base import BaseEmbedding

class BGEM3Embedding(BaseEmbedding):
    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

### Thêm Retrieval Strategy

```python
# Thêm vào src/retrieval/query_expansion.py hoặc file mới
# Sau đó tích hợp vào MultiStrategyRetriever.retrieve_sync()
```

### Thêm Reranker Strategy

```python
from src.core.base import BaseReranker, RetrievedChunk

class MyReranker(BaseReranker):
    async def rerank(self, query, chunks, top_k=50) -> list[RetrievedChunk]:
        ...
```

---

## 7. Optimization Targets

### Recall-First (F2-Macro optimization)
- [x] Multi-query expansion: 2-4 query variations
- [x] HyDE: hypothetical document for better recall
- [x] Large initial retrieval: top-500 → top-1000
- [x] RRF fusion: combine multiple strategies

### Precision Control
- [x] Score threshold >0.95 for high-precision context
- [x] Two-stage reranking (cross-encoder + LLM)
- [x] Score calibration (softmax + minmax)

### Generation Quality
- [x] Legal-specific system prompt
- [x] Structure-aware context formatting
- [x] 2-round self-correction loop
- [x] Citation extraction and validation

---

## 8. Sequence Diagram

```
User     Pipeline      Classifier    Expander     Retriever      Reranker    Generator
 │          │              │            │             │              │           │
 │  query   │              │            │             │              │           │
 │─────────▶│              │            │             │              │           │
 │          │ classify(q)  │            │             │              │           │
 │          │─────────────▶│            │             │              │           │
 │          │◀─────────────┤ qtype      │             │              │           │
 │          │              │            │             │              │           │
 │          │ expand(q)    │            │             │              │           │
 │          │──────────────────────────▶│             │              │           │
 │          │◀──────────────────────────┤ variations  │              │           │
 │          │              │            │             │              │           │
 │          │ embed(q)     │            │             │              │           │
 │          │────────────────────────────────────────▶│              │           │
 │          │◀────────────────────────────────────────┤ emb          │           │
 │          │              │            │             │              │           │
 │          │ multi_search │            │             │              │           │
 │          │────────────────────────────────────────▶│              │           │
 │          │◀────────────────────────────────────────┤ chunks(N)    │           │
 │          │              │            │             │              │           │
 │          │ rerank(q, chunks)         │             │              │           │
 │          │───────────────────────────────────────────────────────▶│           │
 │          │◀───────────────────────────────────────────────────────┤ top-K     │
 │          │              │            │             │              │           │
 │          │ generate(q, context)      │             │              │           │
 │          │───────────────────────────────────────────────────────────────────▶│
 │          │◀───────────────────────────────────────────────────────────────────┤ answer
 │          │              │            │             │              │           │
 │          │ self_correct │            │             │              │           │
 │          │───────────────────────────────────────────────────────────────────▶│
 │          │◀───────────────────────────────────────────────────────────────────┤ corrected
 │          │              │            │             │              │           │
 │◀─────────┤ result       │            │             │              │           │
```

---

## 9. Dependencies

```
Python 3.11+
├── google-genai         # Gemini API
├── torch                # Model inference
├── transformers         # HuggingFace models
├── faiss-cpu            # Dense index
├── rank-bm25            # Sparse index
├── numpy                # Numerical ops
├── scikit-learn         # Metrics
├── scipy                # Softmax
├── datasets             # HuggingFace datasets
├── pydantic-settings    # Configuration
└── tqdm                 # Progress bars
```

---

*Document version: 0.2.0 — Last updated: 2026-06-17*
