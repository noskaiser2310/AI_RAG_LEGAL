import logging
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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

Hãy trả lời câu hỏi dựa trên các điều luật được cung cấp. Trích dẫn điều luật cụ thể."""


class Qwen3Client(BaseLLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading {model_name} with vLLM (Native FP16/BF16)...")
        t0 = time.time()

        # Khởi tạo vLLM engine offline
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192, # Giới hạn 8K cho RAG pháp lý để không ăn hết VRAM, Qwen3 hỗ trợ lên tới 32K native
            enable_reasoning=True, # Bật support reasoning parser của vLLM
            reasoning_parser="qwen3" # Hỗ trợ bóc tách think của qwen3
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info(f"Loaded in {time.time()-t0:.1f}s")

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
        
        # Lấy giá trị enable_thinking từ kwargs, mặc định False theo yêu cầu user
        enable_thinking = kwargs.get("enable_thinking", False)
        
        prompt = self.tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        max_new = kwargs.get("max_tokens", self.max_new_tokens)
        
        # Cấu hình Sampling theo chuẩn tài liệu Qwen3
        if enable_thinking:
            temperature = 0.6
            top_p = 0.95
        else:
            temperature = 0.7
            top_p = 0.8
            
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            max_tokens=max_new,
            repetition_penalty=1.05,
            skip_special_tokens=False, # Giữ lại để xử lý thẻ think nếu cần
        )

        # Suy luận siêu tốc qua vLLM
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]
        
        generated_text = output.outputs[0].text
        
        # Nếu đang bật thinking, vLLM với reasoning_parser="qwen3" sẽ tự động xử lý.
        # Tuy nhiên ta cứ filter thẻ <think> thủ công bằng tay một lần nữa cho chắc ăn nếu nó vẫn lọt ra text
        final_content = generated_text
        if "<think>" in final_content:
            try:
                # Xóa từ <think> đến </think>
                start_idx = final_content.find("<think>")
                end_idx = final_content.find("</think>") + len("</think>")
                if end_idx > start_idx:
                    # Nếu muốn in logic ra console
                    # thinking = final_content[start_idx:end_idx]
                    # print(f"--- Qwen Thinking ---\n{thinking}\n---------------------")
                    
                    final_content = final_content[:start_idx] + final_content[end_idx:]
                    final_content = final_content.strip()
            except Exception as e:
                logger.warning(f"Error parsing <think> block: {e}")

        return LLMResponse(
            text=final_content,
            tokens_input=len(output.prompt_token_ids),
            tokens_output=len(output.outputs[0].token_ids),
            model=self.model_name,
        )

    async def generate_stream(self, messages: list[Message], **kwargs) -> str:
        # vLLM có hỗ trợ AsyncLLMEngine cho streaming, nhưng bản LLM() offline đồng bộ
        # ta tạm trả về kết quả 1 cục. Nếu cần streaming thật, có thể dựng Async Engine
        return (await self.generate(messages, **kwargs)).text
