import logging

import torch
from transformers import AutoModel, AutoTokenizer

from src.core.base import BaseEmbedding
from src.core.config import config

logger = logging.getLogger(__name__)


class HarrierEmbedding(BaseEmbedding):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = config.EMBEDDING_DEVICE or config.DEVICE
        logger.info(f"Loading Harrier: {self.model_name} on {self.device}")
        from src.core.cache import resolve_model_path

        model_path = resolve_model_path(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=model_path != self.model_name
        )
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype,
            local_files_only=model_path != self.model_name,
        ).to(self.device)
        self.model.eval()
        self._dim = config.EMBEDDING_DIM
        self.max_length = 512
        logger.info(f"Harrier loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Last-token pooling cho model decoder (Qwen3). Hidden state token cuối
        (non-pad) tổng hợp toàn bộ ngữ cảnh nhờ causal attention. Xử lý đúng cả
        left-padding lẫn right-padding."""
        left_padded = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
        if left_padded:
            return last_hidden_state[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_idx, seq_lengths]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().tolist()
