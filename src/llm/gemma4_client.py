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


class Gemma4Client(BaseLLM):
    """
    Client for interacting with Google's Gemma models via the new google-genai SDK.
    Supports asynchronous text generation with automatic retries on transient errors.
    """
    def __init__(self, model: str | None = None):
        """
        Initializes the Gemma4 client.
        
        Args:
            model (str, optional): The model ID to use. Defaults to "gemma-4-26b-a4b-it".
        """
        self.model = model or config.GEMINI_MODEL
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        logger.info(f"Gemma4Client initialized with model={self.model}")

    async def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        contents = self._to_gemini_messages(messages)
        system = None
        if contents and contents[0].role == "user":
            system = self._extract_system(messages)

        for attempt in range(MAX_RETRIES):
            try:
                def _do_generate():
                    return self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system,
                            temperature=kwargs.get("temperature", config.TEMPERATURE),
                            max_output_tokens=kwargs.get("max_tokens", config.MAX_TOKENS),
                        ),
                    )
                logger.info(f"Calling Gemini API (attempt {attempt+1}/{MAX_RETRIES})...")
                response = await asyncio.to_thread(_do_generate)
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
                retryable = any(s in err_str for s in (
                    "429", "RESOURCE_EXHAUSTED",      # rate limit
                    "500", "INTERNAL",               # transient server error
                    "503", "UNAVAILABLE", "overloaded",  # service unavailable
                ))
                if retryable and attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(f"Transient error (attempt {attempt+1}/{MAX_RETRIES}), retrying in {wait:.1f}s: {err_str[:80]}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gemma4 generation failed: {e}")
                    return LLMResponse(text=f"Lỗi: {str(e)}", model=self.model)

        return LLMResponse(text="Lỗi: exceeded max retries", model=self.model)

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
