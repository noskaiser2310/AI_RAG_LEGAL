import logging

from src.core.base import BaseLLM, BaseEmbedding, Message

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Bạn là chuyên gia pháp lý Việt Nam. Hãy viết một đoạn văn bản pháp luật mẫu có nội dung liên quan đến câu hỏi sau.

Đoạn văn bản nên:
1. Có cấu trúc như một điều luật thực tế (bao gồm "Điều...")
2. Đề cập đến các khái niệm pháp lý chính trong câu hỏi
3. Có nội dung giải thích hoặc quy định về vấn đề được hỏi

Câu hỏi: {query}

Chỉ trả về đoạn văn bản mẫu, không kèm giải thích."""


class HyDEGenerator:
    def __init__(self, llm: BaseLLM, embedder: BaseEmbedding):
        self.llm = llm
        self.embedder = embedder

    async def generate_hypothetical(self, query: str) -> tuple[str, list[float]]:
        prompt = HYDE_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.7)
        hypo_doc = response.text.strip()
        emb = await self.embedder.embed([hypo_doc])
        logger.info(f"HyDE generated document ({len(hypo_doc)} chars) for query")
        return hypo_doc, emb[0]
