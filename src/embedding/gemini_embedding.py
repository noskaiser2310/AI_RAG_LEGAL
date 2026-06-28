import asyncio
import logging
import time

from google import genai

from src.core.base import BaseEmbedding
from src.core.config import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0


class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str | None = None):
        self.model = model or config.EMBEDDING_MODEL
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self._dim = config.EMBEDDING_DIM
        logger.info(f"GeminiEmbedding: model={self.model} dim={self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(f"Rate limited (attempt {attempt+1}), retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"Embedding failed: {e}")
                    raise

        logger.error("Embedding: exceeded max retries")
        return [[0.0] * self._dim for _ in texts]
