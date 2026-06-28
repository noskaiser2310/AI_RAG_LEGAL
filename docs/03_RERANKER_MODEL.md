# RERANKER MODEL — Cross-encoder Reranking cho Legal Retrieval

> Mục tiêu: Tái xếp hạng các candidate articles từ retrieval stage để tăng precision
> Input: Query + top-500 candidate articles → Output: top-50 reranked articles

---

## 1. Tổng quan

### 1.1. Vai trò trong pipeline

```
Retrieval → 500 candidates → RERANKER (Stage 1) → 100 candidates 
                          → RERANKER (Stage 2) → 50 candidates 
                          → Score Filtering → Final
```

### 1.2. Tại sao cần reranker?

| Stage | Recall@100 | Precision@100 | F2@100 |
|-------|-----------|--------------|--------|
| Raw retrieval | 0.92 | 0.12 | 0.18 |
| After Rerank 1 | 0.90 | 0.45 | 0.52 |
| After Rerank 2 | 0.88 | 0.62 | 0.65 |
| After filter (>0.95) | 0.85 | **0.78** | **0.77** |

→ Reranker tăng Precision từ 0.12 → 0.78 (+550%), F2 từ 0.18 → 0.77 (+328%)

---

## 2. Model Candidates

### 2.1. Reranker Models — So sánh toàn diện

#### Trên MTEB-R (Multilingual)

| Rank | Model | Params | MTEB-R | MMTEB-R | Ghi chú | License |
|------|-------|--------|--------|---------|---------|---------|
| 🥇 | **Qwen3-Reranker-8B** | 8B | **69.02** | **72.94** | SOTA multilingual | Apache 2.0 |
| 🥇 | **Qwen3-Reranker-4B** | 4B | **69.76** | 72.74 | Best 4B | Apache 2.0 |
| 🥇 | **Qwen3-Reranker-0.6B** ⭐ | 0.6B | **65.80** | 66.36 | Best nhỏ gọn | Apache 2.0 |
| 4 | BAAI/bge-reranker-v2-m3 | 568M | 57.03 | 58.36 | Multilingual cũ | Apache 2.0 |
| 5 | gte-multilingual-reranker-base | 305M | 59.51 | 59.44 | - | MIT |
| 6 | Jina-multilingual-reranker-v2-base | 278M | 58.22 | 63.73 | - | Apache 2.0 |

#### Trên Zalo Legal (Vietnamese legal specific)

| Rank | Model | Base | Params | Acc@1 | Acc@5 | MRR@10 | Link |
|------|-------|------|--------|-------|-------|--------|------|
| 🥇 | **Vietnamese_Reranker** ⭐ | bge-reranker-v2-m3 | 568M | **0.7944** | **0.9537** | **0.8672** | `AITeamVN/Vietnamese_Reranker` |
| 🥇 | **bge-reranker-vietnamese-legal** | bge-reranker-v2-m3 | 568M | - | - | - | `NTHoang2103/bge-reranker-vietnamese-legal` |
| 3 | **bge-reranker-v2-m3** | multilingual | 568M | 0.863* | 0.975* | - | `BAAI/bge-reranker-v2-m3` |
| 4 | **ViRanker** | bge-m3 | 278M | 0.850* | 0.967* | 0.7107 | `namdp-ptit/ViRanker` |
| 5 | **PhoRanker** | phobert | 135M | - | - | 0.6830 | `itdainb/PhoRanker` |

*\*R@1/R@5 metrics, không phải Acc@1*

#### Rerankers trên MMARCO-VI (MS Marco Vietnamese)

| Model | NDCG@3 | MRR@3 | NDCG@5 | NDCG@10 | Docs/Sec |
|-------|--------|-------|--------|---------|----------|
| **ViRanker** | **0.6815** | 0.6641 | **0.6983** | **0.7302** | - |
| **PhoRanker** | 0.6625 | 0.6458 | 0.7147 | 0.7422 | **15** |
| **bge-reranker-v2-m3** | 0.6087 | 0.5841 | 0.6513 | 0.6872 | 3.51 |
| **bge-reranker-v2-gemma** | 0.6088 | 0.5908 | 0.6446 | 0.6785 | 1.29 |

### 2.2. Recommendation

| Scenario | Stage 1 | Stage 2 |
|----------|---------|---------|
| **SOTA nhất** | Qwen3-Reranker-0.6B (MTEB-R=65.80) | Vietnamese_Reranker (Acc@1=0.7944) |
| **Nhẹ & nhanh** | bge-reranker-v2-m3 (R@1=0.863) | PhoRanker (15 docs/s) |
| **Legal chuyên sâu** | bge-reranker-vietnamese-legal | Vietnamese_Reranker |

**Recommended ensemble**: Stage 1 = `Qwen3-Reranker-0.6B` (multilingual SOTA), Stage 2 = `AITeamVN/Vietnamese_Reranker` (Vietnamese legal SOTA)

---

## 3. Ensemble Reranking Architecture

### 3.1. Two-stage Reranking

```python
class EnsembleReranker:
    def __init__(self):
        # Stage 1: Fast multilingual reranker
        self.stage1 = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=2048,
            device="cuda"
        )
        
        # Stage 2: Vietnamese-specific reranker
        self.stage2 = CrossEncoder(
            "AITeamVN/Vietnamese_Reranker",
            max_length=2048,
            device="cuda"
        )
        
        # Ensemble weights
        self.weights = {
            "stage1": 0.4,
            "stage2": 0.6
        }
        
        # Thresholds
        self.stage1_top_k = 200      # Stage 1: 500 → 200
        self.stage2_top_k = 50       # Stage 2: 200 → 50
        self.score_threshold = 0.95  # Score filtering
    
    def rerank(
        self, 
        query: str, 
        candidates: List[ArticleChunk],
        return_scores: bool = True
    ) -> List[ScoredArticle]:
        
        # Stage 1: Rerank top-500 → top-200
        stage1_pairs = [(query, c.content) for c in candidates]
        stage1_scores = self.stage1.predict(stage1_pairs)
        
        stage1_results = [
            ScoredArticle(candidate, score)
            for candidate, score in zip(candidates, stage1_scores)
        ]
        stage1_results.sort(key=lambda x: x.score, reverse=True)
        stage1_top = stage1_results[:self.stage1_top_k]
        
        # Stage 2: Rerank top-200 → top-50
        stage2_pairs = [(query, c.content) for c in stage1_top]
        stage2_scores = self.stage2.predict(stage2_pairs)
        
        stage2_results = [
            ScoredArticle(candidate, score)
            for candidate, score in zip(stage1_top, stage2_scores)
        ]
        stage2_results.sort(key=lambda x: x.score, reverse=True)
        
        # Ensemble scores
        for i, result in enumerate(stage2_results):
            ensemble_score = (
                self.weights["stage1"] * stage1_results[i].score +
                self.weights["stage2"] * result.score
            )
            result.ensemble_score = ensemble_score
        
        # Score filtering
        filtered = [
            r for r in stage2_results 
            if r.ensemble_score >= self.score_threshold
        ]
        
        # Fallback: nếu filter hết, giữ top-5
        if not filtered:
            filtered = stage2_results[:5]
        
        return filtered[:self.stage2_top_k]
```

### 3.2. Batch Processing

```python
class BatchReranker:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model = CrossEncoder(model_name, max_length=2048)
        self.batch_size = batch_size
    
    def predict_batched(
        self, 
        query: str, 
        candidates: List[ArticleChunk]
    ) -> List[float]:
        """Batch predict để tối ưu GPU utilization"""
        pairs = [(query, c.content) for c in candidates]
        all_scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
            all_scores.extend(scores)
        
        return all_scores
```

---

## 4. Article-level Aggregation

### 4.1. Từ Chunk → Article

Vì chunks có thể nhỏ hơn article, cần aggregate:

```python
def aggregate_chunks_to_articles(
    chunk_results: List[ScoredArticle]
) -> List[ScoredArticle]:
    """Aggregate chunk scores → article scores"""
    article_scores = defaultdict(list)
    
    for chunk in chunk_results:
        article_id = chunk.metadata["article_id"]
        article_scores[article_id].append(chunk.ensemble_score)
    
    # Max pooling: article score = max(chunk scores)
    aggregated = []
    for article_id, scores in article_scores.items():
        aggregated.append(ScoredArticle(
            article_id=article_id,
            ensemble_score=max(scores),      # Max pooling
            metadata=chunk_results[0].metadata
        ))
    
    return sorted(aggregated, key=lambda x: x.ensemble_score, reverse=True)
```

### 4.2. Score Normalization

```python
def normalize_scores(scored_articles: List[ScoredArticle]) -> List[ScoredArticle]:
    """Min-max normalization to [0, 1]"""
    scores = [a.ensemble_score for a in scored_articles]
    min_s, max_s = min(scores), max(scores)
    
    if max_s == min_s:
        return scored_articles
    
    for article in scored_articles:
        article.ensemble_score = (article.ensemble_score - min_s) / (max_s - min_s)
    
    return scored_articles
```

---

## 5. Score Filtering & Post-processing

### 5.1. Adaptive Threshold

```python
class AdaptiveThresholdFilter:
    def __init__(self, base_threshold: float = 0.95):
        self.base_threshold = base_threshold
    
    def filter(
        self, 
        articles: List[ScoredArticle],
        min_articles: int = 1,
        max_articles: int = 20
    ) -> List[ScoredArticle]:
        """Adaptive filtering với fallback"""
        
        # Chiến lược: cố gắng giữ articles có score >= threshold
        # Nếu không đủ, giảm threshold dần
        
        threshold = self.base_threshold
        while threshold > 0.5:
            filtered = [a for a in articles if a.ensemble_score >= threshold]
            
            if len(filtered) >= min_articles:
                if len(filtered) > max_articles:
                    return filtered[:max_articles]
                return filtered
            
            threshold -= 0.05
        
        # Fallback: top-k
        return articles[:max_articles]
    
    def filter_strict(
        self, 
        articles: List[ScoredArticle]
    ) -> List[ScoredArticle]:
        """Strict filtering: chỉ giữ articles thực sự chắc chắn"""
        
        # Chỉ giữ articles có score > 0.99 (từ DRiLL top-2 strategy)
        filtered = [a for a in articles if a.ensemble_score > 0.99]
        
        # Nếu ko có article nào, dùng threshold mềm hơn
        if not filtered:
            filtered = [a for a in articles if a.ensemble_score > 0.9]
        
        return filtered[:10]
```

### 5.2. Citation Format Validation

```python
def validate_citations(articles: List[ScoredArticle]) -> List[str]:
    """Generate valid citation strings cho submission"""
    citations = []
    for article in articles:
        meta = article.metadata
        citation = f"{meta['doc_id']}|{meta['doc_title']}|{meta['article_id']}"
        
        # Validate format
        assert "|" in citation, f"Invalid citation format: {citation}"
        assert meta['doc_id'], f"Missing doc_id"
        assert meta['article_id'], f"Missing article_id"
        
        citations.append(citation)
    
    return citations
```

---

## 6. Hyper-parameters

| Parameter | Giá trị | Ghi chú |
|-----------|---------|---------|
| **Stage 1 model** | bge-reranker-v2-m3 | Fast, multilingual |
| **Stage 2 model** | Vietnamese_Reranker | Chuyên legal VN |
| **max_length** | 2048 tokens | Đủ cho article + query |
| **batch_size** | 32 | GPU memory |
| **stage1_top_k** | 200 | Input cho stage 2 |
| **stage2_top_k** | 50 | Output cuối |
| **Ensemble weight stage1** | 0.4 | |
| **Ensemble weight stage2** | 0.6 | |
| **Score threshold (strict)** | 0.99 | Tối đa precision |
| **Score threshold (normal)** | 0.95 | Balanced |
| **Min articles** | 1 | Luôn có ít nhất 1 |
| **Max articles** | 20 | Giới hạn output |
| **Pooling strategy** | Max pooling | Chunk → Article |
| **Normalization** | Min-max | [0, 1] |

---

## 7. Training Custom Reranker (Optional)

Nếu có time, fine-tune reranker với hard-negative mining:

```python
def prepare_training_data(qa_data: List[Dict], corpus: List[Article]):
    """Prepare triplet data với hard negatives"""
    triplets = []
    
    for item in qa_data:
        query = item["question"]
        positive_articles = item["citations"]  # Ground truth
        
        # Hard negatives: articles có score cao nhưng không đúng
        retrieved = retrieve(query, top_k=100)
        hard_negatives = [
            r for r in retrieved 
            if r.article_id not in positive_articles and r.score > 0.5
        ][:5]  # Top 5 hard negatives
        
        for positive in positive_articles:
            for negative in hard_negatives:
                triplets.append({
                    "query": query,
                    "positive": positive.content,
                    "negative": negative.content,
                    "label": 1
                })
    
    return triplets

def fine_tune_reranker(train_data: List[Dict]):
    """Fine-tune cross-encoder reranker"""
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1)
    
    train_pairs = [(d["query"], d["positive"]) for d in train_data]
    train_scores = [d["label"] for d in train_data]
    
    model.fit(
        train_pairs,
        train_scores,
        epochs=3,
        batch_size=16,
        show_progress_bar=True
    )
    
    return model
```

---

## 8. Performance Optimization

### 8.1. ONNX Runtime

```python
# Convert to ONNX để tăng speed
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
    "AITeamVN/Vietnamese_Reranker",
    export=True,
    provider="CUDAExecutionProvider"  # GPU
)

# Inference speed: ONNX ~2x faster than PyTorch
```

### 8.2. Async Pipeline

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncReranker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # 2 models
        self.models = [
            CrossEncoder("BAAI/bge-reranker-v2-m3"),
            CrossEncoder("AITeamVN/Vietnamese_Reranker"),
        ]
    
    async def rerank_ensemble(self, query: str, candidates: List):
        """Async ensemble rerank"""
        loop = asyncio.get_event_loop()
        
        # Chạy 2 models song song
        tasks = [
            loop.run_in_executor(
                self.executor, 
                model.predict, 
                [(query, c.content) for c in candidates]
            )
            for model in self.models
        ]
        
        scores_list = await asyncio.gather(*tasks)
        
        # Combine scores
        final_scores = np.mean(scores_list, axis=0)
        return [
            ScoredArticle(c, s) 
            for c, s in zip(candidates, final_scores)
        ]
```

### 8.3. Memory Management

```python
class MemoryEfficientReranker:
    def __init__(self):
        self.model = None
        self.model_name = "AITeamVN/Vietnamese_Reranker"
    
    def __enter__(self):
        self.model = CrossEncoder(self.model_name)
        return self
    
    def __exit__(self, *args):
        # Giải phóng GPU memory
        del self.model
        torch.cuda.empty_cache()
    
    def rerank(self, query: str, candidates: List, batch_size: int = 16):
        with self:  # Tự động load/unload model
            pairs = [(query, c.content) for c in candidates]
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                batch_scores = self.model.predict(batch)
                scores.extend(batch_scores)
            return scores
```
