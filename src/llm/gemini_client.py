import asyncio
import logging
import time

from google import genai
from google.genai import types as genai_types

from src.core.base import BaseLLM, LLMResponse, Message
from src.core.config import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0


class GeminiClient(BaseLLM):
    def __init__(self, model: str | None = None):
        self.model = model or config.GEMINI_MODEL
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        logger.info(f"GeminiClient initialized with model={self.model}")

    async def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        contents = self._to_gemini_messages(messages)
        system = None
        if contents and contents[0].role == "user":
            system = self._extract_system(messages)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=kwargs.get("temperature", config.TEMPERATURE),
                        max_output_tokens=kwargs.get("max_tokens", config.MAX_TOKENS),
                    ),
                )
                text = response.text or ""
                usage = response.usage_metadata or None
                return LLMResponse(
                    text=text,
                    tokens_input=usage.prompt_token_count if usage else 0,
                    tokens_output=usage.candidates_token_count if usage else 0,
                    model=self.model,
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(f"Rate limited (attempt {attempt+1}), retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gemini generation failed: {e}")
                    return LLMResponse(text=f"Lỗi: {str(e)}", model=self.model)

        return LLMResponse(text="Lỗi: exceeded max retries due to rate limiting", model=self.model)

    async def generate_stream(self, messages: list[Message], **kwargs) -> str:
        contents = self._to_gemini_messages(messages)
        system = None
        if contents and contents[0].role == "user":
            system = self._extract_system(messages)
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=kwargs.get("temperature", config.TEMPERATURE),
                max_output_tokens=kwargs.get("max_tokens", config.MAX_TOKENS),
            ),
        )
        result = []
        for chunk in response:
            if chunk.text:
                result.append(chunk.text)
        return "".join(result)

    def _to_gemini_messages(self, messages: list[Message]) -> list[genai_types.Content]:
        result = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role in ("assistant", "model") else "user"
            result.append(genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=m.content)],
            ))
        return result

    def _extract_system(self, messages: list[Message]) -> str | None:
        for m in messages:
            if m.role == "system":
                return m.content
        return None
