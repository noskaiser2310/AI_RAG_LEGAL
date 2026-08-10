import logging

import numpy as np
import torch
from scipy.special import softmax
from sklearn.preprocessing import minmax_scale
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.base import BaseReranker, RetrievedChunk
from src.core.config import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str | None = None, use_softmax: bool = True):
        self.model_name = model_name or config.RERANKER_MODEL
        self.device = config.RERANKER_DEVICE or config.DEVICE
        self.use_softmax = use_softmax
        logger.info(f"Loading reranker: {self.model_name} on {self.device}")
        from transformers import AutoTokenizer
        from src.core.cache import resolve_model_path

        model_path = resolve_model_path(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
            local_files_only=model_path != self.model_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=model_path != self.model_name,
        ).to(self.device)
        self.model.eval()

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 50
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = []
        for c in chunks:
            text = c.content
            if len(text) > 2048:
                text = text[:2048]
                last_space = text.rfind(" ")
                if last_space > 1800:
                    text = text[:last_space]
            pairs.append((query, text))
        scores = self._score_pairs(pairs)
        scores = self._calibrate_scores(scores)

        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
            c.score = float(s)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    def _score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> np.ndarray:
        all_logits = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = np.array([logits.item()])
                all_logits.append(logits)
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logits = np.concatenate(all_logits) if all_logits else np.array([])

        if logits.ndim > 1:
            logits = logits.flatten()

        if self.use_softmax:
            scores = softmax(np.column_stack([np.zeros_like(logits), logits]), axis=1)[:, 1]
        else:
            scores = logits

        return scores

    def _calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) <= 1:
            return scores
        scores = minmax_scale(scores)
        return scores


class LLMReranker(BaseReranker):
    def __init__(self, llm, batch_size: int = 15):
        self.llm = llm
        self.batch_size = batch_size

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 30
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        scored = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            scores = await self._rank_batch(query, batch)
            scored.extend(zip(batch, scores))

        for c, s in scored:
            c.score = s
            c.rerank_score = s

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    async def _rank_batch(self, query: str, batch: list[RetrievedChunk]) -> list[float]:
        items = "\n\n".join(
            f"[{i+1}] {c.content[:800]}"
            for i, c in enumerate(batch)
        )
        prompt = (
            f"Câu hỏi pháp luật: {query}\n\n"
            f"Dưới đây là các đoạn văn bản pháp luật. Hãy chọn số thứ tự của các đoạn "
            f"có liên quan trực tiếp đến câu hỏi (có thể chọn nhiều).\n\n"
            f"{items}\n\n"
            f"Chỉ trả về danh sách số thứ tự, cách nhau bằng dấu phẩy. "
            f"Ví dụ: 1, 3, 5"
        )
        from src.core.base import Message
        resp = await self.llm.generate(
            [Message(role="user", content=prompt)], temperature=0.0
        )

        selected = set()
        for token in resp.text.strip().split(","):
            try:
                idx = int(token.strip()) - 1
                if 0 <= idx < len(batch):
                    selected.add(idx)
            except ValueError:
                pass

        return [1.0 if i in selected else 0.0 for i in range(len(batch))]


class TwoStageReranker(BaseReranker):
    def __init__(self, cross_encoder: CrossEncoderReranker, llm_reranker: LLMReranker):
        self.cross_encoder = cross_encoder
        self.llm_reranker = llm_reranker

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 50
    ) -> list[RetrievedChunk]:
        stage1 = await self.cross_encoder.rerank(query, chunks, top_k=top_k * 2)
        stage2 = await self.llm_reranker.rerank(query, stage1, top_k=top_k)
        return stage2
