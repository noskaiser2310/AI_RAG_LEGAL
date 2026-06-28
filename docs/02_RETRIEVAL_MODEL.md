# RETRIEVAL MODEL — Hệ thống truy hồi văn bản pháp luật

> Mục tiêu: Truy hồi chính xác các điều luật liên quan đến câu hỏi pháp lý
> Primary metric: F2-Macro (Precision + Recall với β=2)

---

## 1. Tổng quan kiến trúc

```
                    Question
                        │
                        ▼
            ┌─────────────────────┐
            │   QUERY EXPANSION   │
            │   Multi-Query (5)   │
            │   HyDE              │
            │   Step-back         │
            └────────┬────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ DENSE    │ │ DENSE    │ │ SPARSE   │
   │ Ensemble │ │ HyDE     │ │ BM25     │
   │ (3 emb)  │ │ Retrieval│ │ (Okapi)  │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
            ┌─────────────────────┐
            │    RRF FUSION       │
            │  top 500 candidates │
            └────────┬────────────┘
                     ▼
            ┌─────────────────────┐
            │   RERANK (x2)       │
            │  → top 100          │
            │  → top 50           │
            └────────┬────────────┘
                     ▼
            ┌─────────────────────┐
            │   SCORE FILTERING   │
            │  threshold > 0.95   │
            └────────┬────────────┘
                     ▼
            Final Relevant Articles
```

---

## 2. Embedding Models (Dense Retrieval)

### 2.1. Model Candidates — Ranking trên Zalo Legal Benchmark

| Rank | Model | Base | Params | Dim | NDCG@10 | MRR@10 | Recall@10 | Link |
|------|-------|------|--------|-----|---------|--------|-----------|------|
| 🥇 | **vietlegal-harrier-0.6b** ⭐ | Harrier (Qwen3) | 600M | 1024 | **0.7813** | **0.7303** | **0.9321** | `mainguyen9/vietlegal-harrier-0.6b` |
| 🥈 | cotu-legal-retriever-Qwen3-Embedding-8B | Qwen3-Embedding-8B | 8B | 4096 | _N/A_ | _N/A_ | _N/A_ | `minhnguyent546/cotu-legal-retriever-Qwen3-Embedding-8B-stage1` |
| 🥉 | cotu-legal-retriever-Qwen3-Embedding-4B | Qwen3-Embedding-4B | 4B | 2560 | _N/A_ | _N/A_ | _N/A_ | `minhnguyent546/cotu-legal-retriever-Qwen3-Embedding-4B-stage1` |
| 4 | **vietlegal-e5** | mE5-large | 560M | 1024 | 0.7310 | 0.6770 | 0.8972 | `mainguyen9/vietlegal-e5` |
| 5 | **AITeamVN/Vietnamese_Embedding_v2** | bge-m3 | 568M | 1024 | 0.7262 | 0.8149* | 0.9578* | `AITeamVN/Vietnamese_Embedding_v2` |
| 6 | **VN Legal Embeddings** | DEk21 | 278M | 768 | 0.8020* | 0.7557* | - | `Quockhanh05/Vietnam_legal_embeddings` |
| 7 | **bge-m3** | multilingual | 568M | 1024 | 0.6660 | 0.6822* | 0.8921* | `BAAI/bge-m3` |
| 8 | **GreenNode-Embedding-Large-VN** | bge-m3 | 568M | 1024 | 0.4794* | - | - | `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1` |

*\*Metrics từ benchmark khác (Zalo Legal Acc/MMARCO), không cùng test set*

**NEW SOTA**: `vietlegal-harrier-0.6b` đạt NDCG@10 = **0.7813** — vượt qua vietlegal-e5 (+5.0 điểm) và harrier gốc (+6.0 điểm)

### 2.2. Ensemble Strategy

```python
class EnsembleEmbedding:
    def __init__(self):
        self.models = {
            "vietlegal_harrier": SentenceTransformer("mainguyen9/vietlegal-harrier-0.6b"),  # SOTA legal VN
            "vietlegal_e5": SentenceTransformer("mainguyen9/vietlegal-e5"),                  # Strong backup
            "qwen3_legal": SentenceTransformer("minhnguyent546/cotu-legal-retriever-Qwen3-Embedding-4B-stage1"),  # Big model
        }
        self.weights = {
            "vietlegal_harrier": 0.5,   # Primary SOTA
            "vietlegal_e5": 0.25,       # Backup legal
            "qwen3_legal": 0.25,        # Large contextual
        }
    
    def encode(self, texts: List[str], query: bool = False) -> np.ndarray:
        """Ensemble embedding với weighted average"""
        if query:
            # Query-side prefix
            text_prefixes = {
                "vn_legal": "query: ",
                "vn_general": "",
                "multilingual": "query: "
            }
        else:
            text_prefixes = {
                "vn_legal": "passage: ",
                "vn_general": "",
                "multilingual": ""
            }
        
        all_embeddings = []
        for name, model in self.models.items():
            prefixed = [text_prefixes[name] + t for t in texts]
            emb = model.encode(prefixed, normalize_embeddings=True)
            all_embeddings.append(emb * self.weights[name])
        
        return np.sum(all_embeddings, axis=0)
```

### 2.3. Search Parameters

| Parameter | Value | Lý do |
|-----------|-------|-------|
| Similarity | Cosine (Inner Product) | Chuẩn cho embedding |
| Index | FAISS IndexFlatIP | Exact search (brute-force) |
| top_k dense | 1000 | Đảm bảo recall |
| Batch size encode | 32 | VRAM efficiency |
| Normalize | always | Cosine ≈ Inner Product |

---

## 3. Sparse Retrieval (BM25)

### 3.1. Configuration

```python
class LegalBM25:
    def __init__(self, corpus: List[str]):
        # Vietnamese tokenizer
        self.tokenizer = VietnameseTokenizer()
        
        # BM25 Okapi parameters
        self.bm25 = BM25Okapi(
            corpus=[self.tokenize(doc) for doc in corpus],
            k1=1.5,    # Term frequency saturation
            b=0.75,    # Length normalization
            epsilon=0.25
        )
    
    def tokenize(self, text: str) -> List[str]:
        """Vietnamese-specific tokenization cho BM25"""
        steps = [
            ("lowercase", str.lower),
            ("word_segment", underthesea.word_tokenize),  # Tách từ tiếng Việt
            ("remove_stopwords", remove_legal_stopwords),  # custom stopwords
            ("extract_articles", extract_article_refs),    # Giữ nguyên "điều_4"
        ]
        for name, func in steps:
            text = func(text)
        return text
    
    def search(self, query: str, top_k: int = 500) -> List[Tuple[str, float]]:
        tokenized = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus[i], scores[i]) for i in top_indices]
```

### 3.2. Vietnamese Tokenizer cho Legal Text

```python
class VietnameseTokenizer:
    def __init__(self):
        # Underthesea cho word segmentation
        # Thêm legal-specific vocabulary
        self.legal_terms = {
            "điều": "điều", "khoản": "khoản", "điểm": "điểm",
            "nghị_định": "nghị_định", "thông_tư": "thông_tư",
            "luật": "luật", "bộ_luật": "bộ_luật",
            # Luật numbers
            "04/2017/qh14": "04/2017/qh14",
        }
    
    def __call__(self, text: str) -> List[str]:
        # 1. Underthesea word segmentation
        words = underthesea.word_tokenize(text)
        
        # 2. Bảo toàn legal entities
        for i, w in enumerate(words):
            if w.lower() in self.legal_terms:
                words[i] = self.legal_terms[w.lower()]
        
        # 3. Bảo toàn article references: "Điều 4" → "điều_4"
        words = self.preserve_article_refs(words)
        
        return words
```

---

## 4. Query Expansion Strategies

### 4.1. Multi-Query Generation

```python
def generate_multi_queries(question: str, n: int = 5) -> List[str]:
    """Sinh N biến thể của câu hỏi"""
    prompts = [
        f"Viết lại câu hỏi pháp lý sau với cách diễn đạt khác:\n{question}",
        f"Hãy liệt kê các từ khóa pháp lý quan trọng trong câu hỏi:\n{question}",
        f"Phân tích câu hỏi pháp lý này thành các khía cạnh pháp lý:\n{question}",
        f"Đặt câu hỏi pháp lý này ở dạng ngắn gọn, dùng từ khóa:\n{question}",
        f"Viết câu hỏi pháp lý này như một luật sư đang hỏi:\n{question}",
    ]
    
    queries = [question]  # Keep original
    for prompt in prompts:
        variant = llm.generate(prompt, max_tokens=100, temperature=0.7)
        queries.append(variant.strip())
    
    return queries[:n]
```

### 4.2. HyDE (Hypothetical Document Embeddings)

```python
def generate_hyde_doc(question: str, domain: str) -> str:
    """Sinh hypothetical legal document answer"""
    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Hãy viết một câu trả lời pháp lý mẫu 
cho câu hỏi sau. Câu trả lời cần viện dẫn điều luật cụ thể và có trích dẫn:

Lĩnh vực: {domain}
Câu hỏi: {question}

Câu trả lời mẫu (sử dụng ngôn ngữ pháp lý chính xác):"""
    
    hyde_doc = llm.generate(prompt, max_tokens=300, temperature=0.3)
    return hyde_doc
```

### 4.3. Step-back Prompting

```python
def generate_stepback_query(question: str) -> str:
    """Sinh câu hỏi tổng quát hơn"""
    prompt = f"""Dựa trên câu hỏi pháp lý sau, hãy viết một câu hỏi tổng quát hơn 
về nguyên tắc pháp lý hoặc quy định chung liên quan:

Câu hỏi: {question}

Câu hỏi tổng quát:"""
    
    stepback = llm.generate(prompt, max_tokens=100, temperature=0.5)
    return stepback.strip()
```

### 4.4. Decomposition (cho câu hỏi phức tạp)

```python
def decompose_question(question: str) -> List[str]:
    """Tách câu hỏi phức tạp thành các câu hỏi con"""
    prompt = f"""Phân tích câu hỏi pháp lý sau thành các câu hỏi con đơn giản hơn:
    
Câu hỏi: {question}

Các câu hỏi con (mỗi câu trên một dòng, bắt đầu bằng -):"""
    
    result = llm.generate(prompt, max_tokens=200, temperature=0.3)
    sub_questions = [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
    return sub_questions
```

---

## 5. Fusion Strategy

### 5.1. Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]], 
    k: int = 60,
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """RRF fusion với optional weights"""
    
    score_dict = defaultdict(float)
    
    for result_idx, results in enumerate(result_lists):
        weight = weights[result_idx] if weights else 1.0
        for rank, (doc_id, _) in enumerate(results):
            # RRF score = weight * 1 / (k + rank)
            score_dict[doc_id] += weight * 1.0 / (k + rank)
    
    # Sort by score
    sorted_docs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

### 5.2. Weighted Hybrid Fusion (Dense + Sparse)

```python
def hybrid_fusion(
    dense_results: List[Tuple[str, float]], 
    sparse_results: List[Tuple[str, float]],
    lambda_weight: float = 0.3
) -> List[Tuple[str, float]]:
    """Weighted combination of dense and sparse scores"""
    
    # Normalize scores to [0, 1]
    dense_scores = normalize_scores(dense_results)
    sparse_scores = normalize_scores(sparse_results)
    
    # Weighted combination
    hybrid_scores = defaultdict(float)
    all_docs = set()
    
    for doc_id, score in dense_scores:
        hybrid_scores[doc_id] += (1 - lambda_weight) * score
        all_docs.add(doc_id)
    
    for doc_id, score in sparse_scores:
        hybrid_scores[doc_id] += lambda_weight * score
        all_docs.add(doc_id)
    
    return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
```

### 5.3. Fusion Pipeline

```python
class RetrievalFusion:
    def __init__(self):
        self.lambda_hybrid = 0.3      # BM25 weight
        self.rrf_k = 60               # RRF constant
        self.strategy_weights = {      # Weights cho mỗi strategy
            "original_dense": 0.4,
            "original_sparse": 0.3,
            "multi_query": 0.15,
            "hyde": 0.1,
            "stepback": 0.05,
        }
    
    def fuse(self, query: str) -> List[Tuple[str, float]]:
        # 1. Expansions
        queries = [query] + generate_multi_queries(query)
        hyde_doc = generate_hyde_doc(query, "general")
        stepback = generate_stepback_query(query)
        
        all_queries = {
            "original": [query],
            "multi_query": queries[1:],
            "hyde": [hyde_doc],
            "stepback": [stepback],
        }
        
        # 2. Retrieve for each strategy
        all_results = []
        for strategy, qs in all_queries.items():
            for q in qs:
                dense = dense_search(q, top_k=500)
                sparse = bm25_search(q, top_k=500)
                hybrid = hybrid_fusion(dense, sparse, self.lambda_hybrid)
                all_results.append(hybrid)
        
        # 3. RRF Fusion
        weights = [
            self.strategy_weights["original_dense"],
            self.strategy_weights["original_sparse"],
        ] + [self.strategy_weights["multi_query"]] * len(queries[1:]) * 2
        weights += [self.strategy_weights["hyde"]] * 2
        weights += [self.strategy_weights["stepback"]] * 2
        
        final = reciprocal_rank_fusion(all_results, k=self.rrf_k, weights=weights)
        return final[:500]
```

---

## 6. Hyper-parameters

| Parameter | Giá trị | Tuning range | Ghi chú |
|-----------|---------|-------------|---------|
| **Embedding ensemble weights** | [0.5, 0.25, 0.25] | [0-1] step 0.1 | Grid search |
| **BM25 k1** | 1.5 | [1.2, 2.0] | Term sat |
| **BM25 b** | 0.75 | [0.5, 1.0] | Length norm |
| **λ (BM25 weight)** | 0.3 | [0.1, 0.5] | Hybrid balance |
| **RRF k** | 60 | [30, 100] | Fusion const |
| **Multi-query variants** | 5 | [3, 10] | |
| **top_k dense** | 1000 | [500, 2000] | |
| **top_k sparse** | 500 | [200, 1000] | |
| **top_k hybrid** | 500 | [200, 1000] | Input cho rerank |
| **Chunk type cho dense** | semantic | [article, semantic, long] | |
| **Chunk type cho sparse** | long | [article, semantic, long] | |

---

## 7. Evaluation Metrics

### 7.1. Metrics cho Retrieval

```python
def calculate_f2(precision: float, recall: float, beta: float = 2.0) -> float:
    """F2 = (1 + β²) × P × R / (β² × P + R)"""
    beta_sq = beta ** 2
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

def macro_f2(queries_results: List[Tuple[List[str], List[str]]]) -> float:
    """Macro-average F2: tính cho từng query rồi trung bình"""
    f2_scores = []
    for predicted_articles, ground_truth_articles in queries_results:
        # Precision
        true_positives = len(set(predicted_articles) & set(ground_truth_articles))
        precision = true_positives / len(predicted_articles) if predicted_articles else 0
        recall = true_positives / len(ground_truth_articles) if ground_truth_articles else 0
        f2_scores.append(calculate_f2(precision, recall))
    
    return np.mean(f2_scores)

def extract_articles_from_answer(answer: str) -> List[str]:
    """Extract 'Điều X' patterns from answer text"""
    pattern = r'Điều\s+\d+[A-Z]?(?:,\s*Điều\s+\d+[A-Z]?)*'
    matches = re.findall(pattern, answer)
    articles = []
    for match in matches:
        articles.extend(re.findall(r'Điều\s+\d+[A-Z]?', match))
    return list(set(articles))
```

### 7.2. Validation Protocol

```python
# Validation with PBGDPL Q&A (4,593 pairs)
def validate_retrieval(pipeline, qa_data):
    results = []
    for item in qa_data:
        query = item["question"]
        ground_truth = extract_cited_articles(item["answer"])
        
        predicted = pipeline.run(query)
        predicted_articles = extract_article_ids(predicted)
        
        f2 = macro_f2([(predicted_articles, ground_truth)])
        results.append(f2)
    
    return {
        "mean_f2": np.mean(results),
        "std_f2": np.std(results),
        "f2_by_domain": group_by_domain(results, qa_data),
    }
```

---

## 8. Implementation Notes

### 8.1. FAISS Index Building

```python
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray, ids: List[int]):
    """Build FAISS index for cosine similarity"""
    dim = embeddings.shape[1]
    
    # Normalize for cosine
    faiss.normalize_L2(embeddings)
    
    # IVF with HNSW for faster search
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
    
    # Train and add
    index.train(embeddings)
    index.add_with_ids(embeddings, np.array(ids))
    
    # Wrap with IDMap for metadata lookup
    index = faiss.IndexIDMap(index)
    
    return index

def search_faiss(index, query_emb: np.ndarray, top_k: int = 100):
    """Search FAISS index"""
    faiss.normalize_L2(query_emb.reshape(1, -1))
    scores, indices = index.search(query_emb.reshape(1, -1), top_k)
    return indices[0], scores[0]
```

### 8.2. Caching Layer

```python
from functools import lru_cache
import hashlib

class RetrievalCache:
    def __init__(self, cache_dir: str = "cache/retrieval"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str, top_k: int) -> Optional[List]:
        cache_path = self.cache_dir / f"{self._hash(query)}_{top_k}.pkl"
        if cache_path.exists():
            return pickle.load(open(cache_path, "rb"))
        return None
    
    def set(self, query: str, top_k: int, results: List):
        cache_path = self.cache_dir / f"{self._hash(query)}_{top_k}.pkl"
        pickle.dump(results, open(cache_path, "wb"))
```

---

## 9. Tối ưu cho F2-Macro

### 9.1. Strategic Optimization

```text
F2 = 5PR / (4P + R)

Since β = 2 → Recall weighs 2× more than Precision

Priority:
1. HIGH RECALL first: Stage 1 (retrieval) → recall ~95%
2. THEN improve precision: Stage 2 (rerank) + filtering
3. FINALLY balance: tune thresholds on validation set
```

### 9.2. Score Filtering Strategy

```python
def optimize_threshold(validation_data, thresholds=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]):
    """Grid search for optimal score threshold"""
    best_f2 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        f2_scores = []
        for query, ground_truth in validation_data:
            results = pipeline.run(query, threshold=threshold)
            predicted = extract_articles(results)
            f2 = macro_f2([(predicted, ground_truth)])
            f2_scores.append(f2)
        
        avg_f2 = np.mean(f2_scores)
        if avg_f2 > best_f2:
            best_f2 = avg_f2
            best_threshold = threshold
    
    return best_threshold, best_f2
```

### 9.3. Test Time Augmentation cho Retrieval

```python
def retrieval_with_tta(query: str, n_augment: int = 3):
    """Test Time Augmentation cho retrieval"""
    # 1. Tạo augmentations
    augmentations = [query]
    for _ in range(n_augment):
        aug = back_translate(query, src="vi", mid="en")  # Vi→En→Vi
        augmentations.append(aug)
    
    # 2. Retrieve for each
    all_results = []
    for aug_query in augmentations:
        results = retrieve(aug_query, top_k=200)
        all_results.append(results)
    
    # 3. RRF fusion
    final = reciprocal_rank_fusion(all_results, k=30)
    return final[:100]
```
