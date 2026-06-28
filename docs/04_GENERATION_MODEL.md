# GENERATION MODEL — LLM cho Legal Question Answering

> Mục tiêu: Sinh câu trả lời pháp lý chính xác, dẫn nguồn điều luật
> Constraints: < 14B params, open-source, phát hành trước 01/03/2026

---

## 1. Tổng quan

### 1.1. Vai trò trong pipeline

```
Retrieved Articles → LEGAL REASONING AGENT → QA GENERATION AGENT → VERIFICATION AGENT → Final Answer
                                              ↓
                                        LLM Ensemble
                                     (Qwen3-4B + Legal VN)
                                              ↓
                                     answer + relevant_docs + relevant_articles
```

### 1.2. Output Requirements (Competition Format)

```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào?",
  "answer": "Doanh nghiệp được hỗ trợ khi đáp ứng tiêu chí... (theo Điều 4, Luật Hỗ trợ DNNVV)",
  "relevant_docs": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết..."
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"
  ]
}
```

---

## 2. LLM Candidates

### 2.1. So sánh các model

| Model | Params | Context | Release | Vietnamese | Legal Reasoning | VRAM (4-bit) |
|-------|--------|---------|---------|------------|----------------|------|
| **Qwen3-8B-Instruct** 🏆 | 8.2B | 128K | 04/2025 | ★★★★ | ★★★★ | ~6GB |
| **Qwen3-4B-Instruct** ⭐ | 4.0B | 128K | 04/2025 | ★★★★ | ★★★★ | ~4GB |
| **thangvip/legal-grpo** ⭐ | 4.0B | 32K | 2025 | ★★★★★ | ★★★★★ | ~4GB |
| **bqbbao6/Qwen2.5-7B-legal-vn** | 7.0B | 32K | 2025 | ★★★★★ | ★★★★★ | ~6GB |
| **VLSP2025/qwen3-4b-legal-pretrain** | 4.0B | 8K | 2025 | ★★★★★ | ★★★★★ | ~4GB |
| **ViLegalQwen3-1.7B-Base** | 1.7B | 32K | 2025 | ★★★★ | ★★★★ | ~2GB |
| **Qwen2.5-7B-Instruct** | 7.6B | 32K | 09/2024 | ★★★★ | ★★★★ | ~6GB |
| **SeaLLM-v3-7B-Chat** | 7.0B | 8K | 2024 | ★★★★ | ★★★ | ~6GB |
| **DeepSeek-R1-Distill-Qwen-7B** | 7.6B | 32K | 01/2025 | ★★★ | ★★★★ | ~6GB |

### 2.2. Model Phân tích chuyên sâu

#### 🏆 Qwen3-8B-Instruct (Primary — Best Overall)
```yaml
name: Qwen/Qwen3-8B-Instruct
params: 8.2B
context: 131072 (128K)
strengths:
  - "Strongest under 14B: beats Qwen2.5-14B, LLaMA-3-8B"
  - "128K context → full document retrieval"
  - "119 languages support → Vietnamese natively strong"
  - "Thinking mode (extended reasoning via special tokens)"
  - "4-bit quantization → ~6GB VRAM"
benchmarks:
  MMLU-Pro: 69.2     # General knowledge
  MMLU-Redux: 84.4
  MGSM: 83.6         # Multilingual reasoning
  MMMLU: 75.4        # Multilingual knowledge
  LiveCodeBench: 47.2
```

#### ⭐ Qwen3-4B-Instruct (Efficient Primary)
```yaml
name: Qwen/Qwen3-4B-Instruct
params: 4.0B
architecture: Qwen3ForCausalLM
context: 131072 (128K)
strengths:
  - "Performance vượt Qwen2.5-7B trên hầu hết benchmarks"
  - "128K context → đủ cho nhiều articles"
  - "Tiếng Việt tốt (support 119 languages)"
  - "Thinking mode (multi-step reasoning)"
  - "Efficient: 4B params = 4GB VRAM (4-bit)"
benchmarks:
  MMLU: 72.99
  MGSM: 67.74   # Multilingual math (reasoning)
  MMMLU: 71.42  # Multilingual knowledge
```

#### 🥇 thangvip/qwen3-4b-vietnamese-legal-grpo (Legal-tuned)
```yaml
name: thangvip/qwen3-4b-vietnamese-legal-grpo
base: qwen3-4b-legal-pretrain-synthetic-8k
params: 4B
training: GRPO (Group Relative Policy Optimization)
domain: Vietnamese legal reasoning
features:
  - "Syllogistic reasoning: Major Premise → Minor Premise → Conclusion"
  - "Citation support with XML tags"
  - "Designed for Vietnamese legal QA"
evaluation_criteria:
  correctness: 0.35
  format_compliance: 0.20
  citation_accuracy: 0.15
  reasoning_quality: 0.15
  hallucination_penalty: 0.10
```

#### 🥈 bqbbao6/Qwen2.5-7B-legal-vn (Legal-tuned 7B)
```yaml
name: bqbbao6/Qwen2.5-7B-legal-vn
base: Qwen2.5-7B-Instruct
params: 7B (4-bit quantized → ~6GB VRAM)
training: QLoRA (Rank 16, Alpha 32)
features:
  - "Strict context adherence (RAG-focused)"
  - "Legal formalism & authoritative tone"
  - "Reduced hallucination for legal questions"
```

#### VLSP2025-LegalSML/qwen3-4b-legal-pretrain
```yaml
name: VLSP2025-LegalSML/qwen3-4b-legal-pretrain
base: Qwen3-4B-Base
params: 4B
training: Continual pretraining + synthetic legal data (8K tokens)
domain: Vietnamese legal corpus
features:
  - "Tiếp tục pretrain trên legal corpus → domain knowledge"
  - "Cần fine-tune instruct trước khi dùng"
  - "Cơ sở cho legal-GRPO models"
```

#### ViLegalQwen3-1.7B-Base
```yaml
name: minhnguyent546/ViLegalQwen3-1.7B-Base
base: Qwen3-1.7B-Base
params: 1.7B
training: Continual pretraining on Vietnamese legal
features:
  - "Siêu nhẹ: 1.7B → suitable for draft model or ensemble phụ"
  - "Cần fine-tune instruct"
  - "Speculative decoding draft candidate"
```

### 2.3. Recommendation

| Scenario | Primary Model | Ensemble Model |
|----------|--------------|---------------|
| **GPU 6GB** | Qwen3-4B-Instruct (4-bit) | thangvip/qwen3-4b-legal-grpo |
| **GPU 12GB** | Qwen3-8B-Instruct (4-bit) | thangvip/qwen3-4b-legal-grpo |
| **GPU 24GB+** | Qwen3-8B-Instruct (full) | Ensemble cả 3 + ViLegalQwen3-1.7B as draft |

**Recommend**: Qwen3-8B-Instruct (primary, 4-bit ~6GB) + thangvip/legal-grpo (backup) → ensemble voting + speculative decoding draft

Note: Qwen3-8B-Instruct ở 4-bit chỉ ~6GB VRAM — hoàn toàn khả thi trên GPU tầm trung, trong khi chất lượng vượt trội Qwen3-4B.

---

## 3. LLM Inference Engine

### 3.1. vLLM Setup

```python
# vLLM server (recommended for production)
from vllm import LLM, SamplingParams

class LegalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct"):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,      # Low temperature = deterministic
            top_p=0.9,
            max_tokens=1024,
            stop=["</s>", "\n\n\n"],
        )
    
    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
```

### 3.2. Quantization

```python
# 4-bit quantization để giảm VRAM
class QuantizedLegalLLM:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,  # Greedy decoding
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 4. Prompt Engineering

### 4.1. System Prompt

```python
SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên hỗ trợ doanh nghiệp SME tại Việt Nam. 
Nhiệm vụ của bạn là trả lời các câu hỏi pháp lý dựa trên các điều luật được cung cấp.

NGUYÊN TẮC:
1. Chỉ trả lời dựa trên các điều luật được cung cấp trong ngữ cảnh
2. Luôn trích dẫn điều luật cụ thể khi trả lời (Điều X, Khoản Y của [Tên văn bản])
3. Nếu thông tin không đầy đủ, nêu rõ điều luật liên quan nhất
4. KHÔNG bịa ra điều luật hoặc thông tin không có trong ngữ cảnh
5. KHÔNG sử dụng thông tin từ kiến thức có sẵn nếu không được cung cấp
6. Trả lời bằng tiếng Việt, ngôn ngữ pháp lý chính xác nhưng dễ hiểu
7. Kết thúc với cảnh báo: "Lưu ý: Đây là thông tin tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.""""
```

### 4.2. Context Template

```python
def format_context(articles: List[ScoredArticle]) -> str:
    """Format retrieved articles vào context cho LLM"""
    context_parts = []
    
    for i, article in enumerate(articles, 1):
        meta = article.metadata
        context_parts.append(f"""
[{i}] Văn bản: {meta['doc_title']} ({meta['doc_id']})
    {meta['article_id']}: {article.content}
""")
    
    return "\n".join(context_parts)
```

### 4.3. Full Prompt Template

```python
def build_prompt(
    question: str,
    articles: List[ScoredArticle],
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    context = format_context(articles)
    
    prompt = f"""{system_prompt}

=== CÁC ĐIỀU LUẬT LIÊN QUAN ===

{context}

=== CÂU HỎI ===

{question}

=== YÊU CẦU ===

Hãy trả lời câu hỏi trên dựa trên các điều luật được cung cấp. 
Câu trả lời cần:
1. Trích dẫn điều luật cụ thể (Điều X, Khoản Y - [Tên văn bản])
2. Giải thích rõ ràng, dễ hiểu
3. Đề cập đến các quyền, nghĩa vụ, điều kiện nếu có
4. Kết thúc với cảnh báo pháp lý

TRẢ LỜI:"""
    
    return prompt
```

### 4.4. Legal Reasoning Prompt

```python
REASONING_PROMPT = """Bạn là chuyên gia phân tích pháp luật. 
Hãy phân tích từng điều luật dưới đây xem có liên quan đến câu hỏi không.

Câu hỏi: {question}

{articles}

Với MỖI điều luật, hãy trả lời:
1. Điều luật này có LIÊN QUAN trực tiếp đến câu hỏi không? (CÓ/KHÔNG)
2. Mức độ liên quan: CAO / TRUNG BÌNH / THẤP
3. Lý do ngắn gọn (1 câu)
4. Nếu CÓ: điều luật này trả lời phần nào của câu hỏi?

Định dạng:
[1] Điều 4 - MỨC: CAO
    Lý do: Điều luật quy định trực tiếp về...
    
[2] Điều 5 - MỨC: TRUNG BÌNH
    Lý do: Liên quan gián tiếp, quy định về...
"""
```

---

## 5. Multi-LLM Ensemble

### 5.1. Voting Strategy

```python
class LLMEnsemble:
    def __init__(self, models: List[LegalLLM]):
        self.models = models
    
    def generate_with_voting(
        self, 
        prompt: str,
        n_attempts: int = 3
    ) -> Dict:
        """Multiple models generate → vote for best answer"""
        all_answers = []
        
        for model in self.models:
            for _ in range(n_attempts):
                answer = model.generate(prompt)
                all_answers.append(answer)
        
        # Self-evaluation: mỗi answer tự chấm điểm
        scored_answers = []
        for answer in all_answers:
            score = self.evaluate_answer(prompt, answer)
            scored_answers.append((answer, score))
        
        # Chọn answer có điểm cao nhất
        best = max(scored_answers, key=lambda x: x[1])
        
        return {
            "answer": best[0],
            "score": best[1],
            "all_answers": [a for a, s in scored_answers],
            "all_scores": [s for a, s in scored_answers],
        }
    
    def evaluate_answer(self, prompt: str, answer: str) -> float:
        """Self-evaluation: LLM tự chấm điểm câu trả lời"""
        eval_prompt = f"""Đánh giá câu trả lời pháp lý sau theo các tiêu chí:
1. Tính chính xác (0-10): Có đúng với điều luật không?
2. Trích dẫn (0-10): Có dẫn nguồn điều luật cụ thể không?
3. Đầy đủ (0-10): Có trả lời đầy đủ câu hỏi không?
4. Rõ ràng (0-10): Có dễ hiểu không?

Câu hỏi: {extract_question(prompt)}
Câu trả lời: {answer}

Điểm (chỉ trả về số trung bình, ví dụ: 8.5):"""
        
        score_text = self.models[0].generate(eval_prompt)
        try:
            return float(score_text.strip())
        except:
            return 7.0  # Default score
```

### 5.2. Speculative Decoding (Optional)

```python
# Dùng small model draft + large model verify
# Qwen3-0.6B draft → Qwen3-4B verify
from vllm import LLM

draft_model = LLM("Qwen/Qwen3-0.6B-Instruct")
target_model = LLM("Qwen/Qwen3-4B-Instruct")

# vLLM supports speculative decoding natively
llm = LLM(
    model="Qwen/Qwen3-4B-Instruct",
    speculative_model="Qwen/Qwen3-0.6B-Instruct",
    num_speculative_tokens=5,
)
```

---

## 6. Output Post-processing

### 6.1. Article Citation Extraction

```python
def extract_citations(answer: str, context_articles: List[ScoredArticle]) -> Tuple[List[str], List[str]]:
    """Extract relevant_docs and relevant_articles từ answer"""
    
    # Pattern: "Điều X" or "Điều X, Khoản Y"
    article_pattern = r'Điều\s+\d+[A-Z]?(?:\s*,\s*Điều\s+\d+[A-Z]?)*'
    doc_pattern = r'(Luật|Nghị định|Thông tư|Quyết định)\s+\d+[^,]*'
    
    found_articles = set()
    found_docs = set()
    
    # Tìm "Điều X" trong answer
    for match in re.finditer(article_pattern, answer):
        articles_in_text = re.findall(r'Điều\s+\d+[A-Z]?', match.group())
        
        for art in articles_in_text:
            # Map to article metadata
            for ctx_art in context_articles:
                if art.lower() in ctx_art.metadata["article_id"].lower():
                    found_articles.add(ctx_art)
                    found_docs.add(ctx_art.metadata["doc_id"])
    
    # Format output
    relevant_docs = []
    relevant_articles = []
    
    for article in found_articles:
        meta = article.metadata
        doc_str = f"{meta['doc_id']}|{meta['doc_title']}"
        art_str = f"{meta['doc_id']}|{meta['doc_title']}|{meta['article_id']}"
        
        if doc_str not in relevant_docs:
            relevant_docs.append(doc_str)
        relevant_articles.append(art_str)
    
    return relevant_docs, relevant_articles
```

### 6.2. Answer Quality Check

```python
class AnswerValidator:
    def __init__(self):
        self.min_citations = 1  # Phải có ít nhất 1 citation
        self.max_citations = 20
        self.min_answer_length = 50  # Characters
        self.max_answer_length = 2000
    
    def validate(self, submission: Dict) -> bool:
        checks = []
        
        # Check length
        answer = submission["answer"]
        checks.append(len(answer) >= self.min_answer_length)
        checks.append(len(answer) <= self.max_answer_length)
        
        # Check citations
        checks.append(len(submission["relevant_articles"]) >= self.min_citations)
        checks.append(len(submission["relevant_articles"]) <= self.max_citations)
        
        # Check format
        for art in submission["relevant_articles"]:
            parts = art.split("|")
            checks.append(len(parts) == 3)  # doc_id|title|article
            checks.append(parts[2].startswith("Điều"))
        
        for doc in submission["relevant_docs"]:
            parts = doc.split("|")
            checks.append(len(parts) == 2)  # doc_id|title
        
        return all(checks)
```

---

## 7. Verification Agent (Self-correction)

### 7.1. Citation Verification

```python
def verify_citations(answer: str, corpus_index: Dict) -> Dict:
    """Verify that cited articles actually exist"""
    citations = re.findall(r'Điều\s+\d+[A-Z]?', answer)
    fake_citations = []
    
    for citation in citations:
        if not article_exists(citation, corpus_index):
            fake_citations.append(citation)
    
    return {
        "valid": len(fake_citations) == 0,
        "fake_citations": fake_citations,
        "total_citations": len(citations),
        "valid_citations": len(citations) - len(fake_citations),
    }
```

### 7.2. Hallucination Detection

```python
def detect_hallucination(question: str, answer: str, context: str) -> Dict:
    """Detect hallucinations using LLM self-check"""
    
    check_prompt = f"""So sánh câu trả lời với ngữ cảnh pháp luật được cung cấp.
    
NGỮ CẢNH:
{context}

CÂU TRẢ LỜI:
{answer}

Hãy kiểm tra:
1. Mọi thông tin trong câu trả lời có được hỗ trợ bởi ngữ cảnh không?
2. Có điều luật nào được trích dẫn nhưng không có trong ngữ cảnh không?
3. Có thông tin nào sai lệch so với ngữ cảnh không?

Kết luận (CHẤP NHẬN / TỪ CHỐI):
Lý do:"""
    
    verdict = llm.generate(check_prompt, max_tokens=100)
    
    return {
        "accepted": "CHẤP NHẬN" in verdict,
        "reason": verdict,
        "needs_correction": "TỪ CHỐI" in verdict,
    }
```

### 7.3. Self-Correction

```python
def self_correct(question: str, articles: List[ScoredArticle], draft_answer: str) -> str:
    """Self-correction loop: generate → evaluate → refine"""
    max_iterations = 3
    current_answer = draft_answer
    
    for i in range(max_iterations):
        # Check for issues
        citation_check = verify_citations(current_answer, articles)
        hallucination_check = detect_hallucination(question, current_answer, articles)
        
        if citation_check["valid"] and hallucination_check["accepted"]:
            break  # Good enough
        
        # Refine
        correction_prompt = f"""Câu trả lời trước có vấn đề:
- Citation issues: {citation_check['fake_citations']}
- Hallucination: {hallucination_check['reason']}

Hãy sửa lại câu trả lời dựa trên ngữ cảnh, chỉ sử dụng thông tin từ các điều luật được cung cấp.

Ngữ cảnh: {format_context(articles)}
Câu hỏi: {question}

Câu trả lời đã sửa:"""
        
        current_answer = llm.generate(correction_prompt)
    
    return current_answer
```

---

## 8. Complete QA Pipeline

```python
class QAPipeline:
    def __init__(self):
        self.llm_primary = LegalLLM("Qwen/Qwen3-4B-Instruct")
        self.llm_legal = LegalLLM("thangvip/qwen3-4b-vietnamese-legal-grpo")
        
    def generate_submission_item(
        self,
        query: LegalQuery,
        articles: List[ScoredArticle]
    ) -> SubmissionItem:
        
        # Step 1: Build prompt
        prompt = build_prompt(query.question, articles)
        
        # Step 2: Multi-LLM generation
        ensemble = LLMEnsemble([self.llm_primary, self.llm_legal])
        result = ensemble.generate_with_voting(prompt)
        
        # Step 3: Self-correction
        final_answer = self_correct(
            query.question, 
            articles, 
            result["answer"]
        )
        
        # Step 4: Extract citations
        relevant_docs, relevant_articles = extract_citations(
            final_answer, 
            articles
        )
        
        # Step 5: Validate
        validator = AnswerValidator()
        item = SubmissionItem(
            id=query.id,
            question=query.question,
            answer=final_answer,
            relevant_docs=relevant_docs,
            relevant_articles=relevant_articles
        )
        
        if not validator.validate(item.__dict__):
            # Fallback: use raw retrieval results
            item.relevant_docs = extract_unique_docs(articles)
            item.relevant_articles = extract_all_articles(articles)
        
        return item
```

---

## 9. Hyper-parameters

| Parameter | Value | Ghi chú |
|-----------|-------|---------|
| **temperature** | 0.1 | Deterministic |
| **top_p** | 0.9 | |
| **max_tokens** | 1024 | Đủ cho answer dài |
| **n_ensemble_models** | 2 | Qwen3-4B + Legal-GRPO |
| **n_attempts_per_model** | 2 | |
| **self_correct_max_iter** | 3 | |
| **min_citations** | 1 | |
| **max_citations** | 20 | |
| **min_answer_chars** | 50 | |
| **max_answer_chars** | 2000 | |
| **score_threshold** | 0.95 | Cho retrieval |
| **fallback_citations** | all retrieved articles | When validation fails |
