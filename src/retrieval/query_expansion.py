import logging

from src.core.base import BaseLLM, BaseQueryExpander, Message

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """Bạn là chuyên gia pháp lý. Hãy mở rộng câu hỏi pháp luật sau đây thành {num_variations} câu hỏi khác nhau.

Mỗi câu hỏi mở rộng nên:
1. Giữ nguyên ý chính pháp lý
2. Sử dụng từ ngữ pháp lý đồng nghĩa hoặc cách diễn đạt khác
3. Có thể đề cập đến các điều luật hoặc văn bản pháp luật liên quan

Câu hỏi gốc: {query}

Trả về mỗi câu hỏi trên một dòng, bắt đầu bằng dấu "- "."""


class LLMQueryExpander(BaseQueryExpander):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def expand(self, query: str, num_variations: int = 3) -> list[str]:
        prompt = EXPANSION_PROMPT.format(query=query, num_variations=num_variations)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.3)
        variations = [
            line.strip().lstrip("- ").strip()
            for line in response.text.strip().split("\n")
            if line.strip().startswith("-")
        ]
        variations = [v for v in variations if v and len(v) > 5]
        all_queries = [query] + variations[:num_variations]
        logger.info(f"Query expansion: {query} -> {len(all_queries)-1} variations")
        return all_queries
