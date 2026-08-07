import logging
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.core.base import BaseLLM, LLMResponse, Message
from src.core.config import config

logger = logging.getLogger(__name__)

QWEN_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên hỗ trợ doanh nghiệp SME tại Việt Nam.
Nhiệm vụ của bạn là trả lời các câu hỏi pháp lý dựa trên các điều luật được cung cấp.

NGUYÊN TẮC:
1. Chỉ trả lời dựa trên các điều luật được cung cấp trong ngữ cảnh
2. Luôn trích dẫn điều luật cụ thể khi trả lời (Điều X, Khoản Y của [Tên văn bản])
3. Nếu thông tin không đầy đủ, nêu rõ điều luật liên quan nhất
4. KHÔNG bịa ra điều luật hoặc thông tin không có trong ngữ cảnh
5. Trả lời bằng tiếng Việt, ngôn ngữ pháp lý chính xác nhưng dễ hiểu
6. Kết thúc với cảnh báo: "Lưu ý: Đây là thông tin tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp."

Hãy trả lời câu hỏi dựa trên các điều luật được cung cấp. Trích dẫn điều luật cụ thể."""  # noqa: E501


class HFClient(BaseLLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading {model_name} {'4-bit' if load_in_4bit else 'full'}...")
        t0 = time.time()

        quant_config = None
        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to fp16")
                load_in_4bit = False
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        logger.info(f"Loaded in {time.time()-t0:.1f}s ({self._count_params():.1f}B)")

    def _count_params(self) -> float:
        try:
            return sum(p.numel() for p in self.model.parameters()) / 1e9
        except Exception:
            return 0

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        has_system = False
        for m in messages:
            if m.role == "system":
                result.append({"role": "system", "content": m.content})
                has_system = True
            elif m.role in ("user", "assistant"):
                result.append({"role": m.role, "content": m.content})
        if not has_system:
            result.insert(0, {"role": "system", "content": QWEN_SYSTEM_PROMPT})
        return result

    async def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        msgs = self._build_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_new = kwargs.get("max_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", config.TEMPERATURE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return LLMResponse(
            text=text,
            tokens_input=inputs.input_ids.shape[1],
            tokens_output=len(generated),
            model=self.model_name,
        )

    async def generate_stream(self, messages: list[Message], **kwargs) -> str:
        return (await self.generate(messages, **kwargs)).text
