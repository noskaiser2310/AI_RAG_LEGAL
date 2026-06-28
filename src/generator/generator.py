import logging

from src.core.base import BaseLLM, Message
from src.core.config import config
from src.retrieval.text_processor import extract_article_references

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên nghiệp và tận tâm chuyên về luật pháp Việt Nam.
Nhiệm vụ của bạn là cung cấp câu trả lời chi tiết, chuyên nghiệp, giải thích rõ ràng và chính xác dựa trên các điều luật được cung cấp.

Hướng dẫn BẮT BUỘC:
1. CHỈ trả lời dựa trên thông tin trong các đoạn văn bản được cung cấp bên dưới.
2. TUYỆT ĐỐI KHÔNG sử dụng ngôn từ giao tiếp không phù hợp (Ví dụ: KHÔNG dùng "Chào bạn", "Dưới đây là câu trả lời", "Theo tôi", "Câu trả lời là"). Đi thẳng vào nội dung pháp lý.
3. LUÔN trích dẫn điều luật cụ thể (ví dụ: "theo Điều 123..."). Nếu không có số điều, ghi rõ tên văn bản.
4. Trả lời bằng tiếng Việt chuyên ngành pháp lý, cấu trúc mạch lạc, phân tích đầy đủ và chuyên sâu, trình bày các ý rõ ràng.
5. XUNG ĐỘT PHÁP LÝ (RẤT QUAN TRỌNG): Nếu tài liệu cung cấp chứa nhiều phiên bản của cùng một luật (ví dụ: Luật Đấu thầu 2013 và Luật Đấu thầu 2023), BẠN PHẢI luôn ưu tiên áp dụng văn bản mới nhất, còn hiệu lực và bỏ qua văn bản cũ, trừ khi người dùng chỉ định rõ năm.
6. Nếu thông tin không đủ, hãy trả lời đúng một câu: "Tôi không tìm thấy đủ thông tin pháp lý để trả lời câu hỏi này." """

SELF_CORRECT_1 = """Bạn là chuyên gia pháp lý thẩm định. Hãy đánh giá và sửa lỗi câu trả lời sau:

Câu hỏi: {query}

Các đoạn văn bản pháp luật được cung cấp:
{context}

Câu trả lời hiện tại: {current_answer}

Nhiệm vụ:
1. Chỉ ra những điểm chưa chính xác, thông tin bị thiếu, hoặc trích dẫn sai lệch so với văn bản pháp luật (nếu có).
2. Dựa trên nhận xét đó, hãy viết lại câu trả lời đầy đủ và chính xác hơn. Bạn có thể giải thích lý do sửa đổi.

Nếu câu trả lời hiện tại đã HOÀN TOÀN CHÍNH XÁC và KHÔNG THỂ TỐT HƠN, hãy trả lời đúng 2 chữ: OK"""

SELF_CORRECT_2 = """Bạn là luật sư cấp cao. Hãy thẩm định lần 2 câu trả lời sau khi đã được chỉnh sửa:

Câu hỏi: {query}

Đoạn văn bản pháp luật:
{context}

Câu trả lời hiện tại:
{current_answer}

Nhiệm vụ:
1. Rà soát xem còn lỗi sai pháp lý, thiếu ý, hoặc trích dẫn sai nào không.
2. Viết lại câu trả lời hoàn thiện nhất dựa trên những điểm cần cải thiện.

Nếu câu trả lời đã HOÀN HẢO, trả lời đúng 2 chữ: OK"""

FINAL_EDITOR_PROMPT = """Bạn là biên tập viên xuất bản chuyên nghiệp. Nhiệm vụ của bạn là chuyển đổi nội dung pháp lý thô sau đây thành CÂU TRẢ LỜI CHUẨN MỰC CUỐI CÙNG với độ chi tiết và tính chuyên nghiệp cao nhất.

Câu hỏi: {query}

Nội dung pháp lý thô (có thể chứa các nhận xét thừa của chuyên gia): 
{raw_answer}

YÊU CẦU BẮT BUỘC (SỐNG CÒN):
1. CHỈ in ra DUY NHẤT nội dung câu trả lời cuối cùng.
2. TUYỆT ĐỐI KHÔNG sử dụng ngôn từ giao tiếp, bình luận, dẫn dắt (Ví dụ: KHÔNG dùng "Chào bạn", "Đây là bản chỉnh sửa", "Đánh giá chuyên gia", "Theo chuyên gia", "Dưới đây là").
3. Trình bày câu trả lời chi tiết, chuyên nghiệp, phân tích đầy đủ các khía cạnh pháp lý của vấn đề. Đảm bảo cấu trúc rõ ràng, logic và dễ hiểu.
4. Giữ lại nguyên vẹn mọi trích dẫn điều luật (ví dụ: "theo Điều 12..."). Không được bỏ sót bất kỳ căn cứ pháp lý nào.
5. XÓA BỎ toàn bộ các chú thích nguồn dạng ngoặc vuông như "[1]", "[2]" trong văn bản.
6. Nếu nội dung thô có chứa các tiêu đề như "Câu trả lời đã sửa", "Đánh giá của tôi", BẮT BUỘC PHẢI XÓA CHÚNG."""

NEGATIVE_DETECTION_PROMPT = """Kiểm tra xem câu trả lời sau có chứa thông tin pháp lý hữu ích không.

Câu hỏi: {query}
Câu trả lời: {answer}

Nếu câu trả lời chứa thông tin pháp lý cụ thể và trả lời trực tiếp câu hỏi, trả lời: USEFUL
Nếu câu trả lời chung chung, không cụ thể hoặc không trả lời câu hỏi, trả lời: NOT_USEFUL"""


def format_context(chunks: list) -> str:
    parts = []
    seen_sections = set()
    for i, c in enumerate(chunks):
        section_key = f"{c.doc_title}|{c.article_id}"
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)
        title = c.doc_title or f"Văn bản"
        article = f" - Điều {c.article_id}" if c.article_id else ""
        parts.append(f"[{i+1}] {title}{article}:\n{c.content[:1500]}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, llm: BaseLLM, retriever=None):
        self.llm = llm
        self.retriever = retriever

    async def generate(
        self,
        query: str,
        chunks: list | None = None,
        use_self_correct: bool = True,
        max_correction_rounds: int = 2,
        score_threshold: float = 0.95,
    ) -> tuple[str, int, list[str]]:
        if chunks is None and self.retriever:
            chunks = await self.retriever.retrieve(query)

        high_conf = [c for c in chunks if c.score >= score_threshold]
        context_chunks = high_conf if len(high_conf) >= 5 else chunks[:config.FINAL_TOP_K]

        context = format_context(context_chunks)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=(
                f"Dựa trên các điều luật sau, hãy trả lời câu hỏi.\n\n"
                f"--- CÁC ĐIỀU LUẬT ---\n{context}\n\n"
                f"--- CÂU HỎI ---\n{query}"
            )),
        ]
        response = await self.llm.generate(messages)
        answer = response.text
        num_corrections = 0

        is_useful = await self._detect_negative(query, answer)
        if not is_useful:
            logger.warning(f"Negative response detected for query: {query[:60]}")
            messages.append(Message(role="model", content=answer))
            messages.append(Message(
                role="user",
                content=(
                    f"Câu trả lời của bạn quá chung chung. Hãy trả lời CỤ THỂ dựa trên "
                    f"các điều luật được cung cấp. Nếu không có thông tin, hãy nói rõ "
                    f"là không tìm thấy thông tin pháp lý phù hợp."
                ),
            ))
            retry = await self.llm.generate(messages)
            answer = retry.text

        if use_self_correct:
            answer, num_corrections = await self._self_correct_loop(
                query, context, answer, max_correction_rounds
            )
            
        # Thêm bước Biên tập cuối cùng (Final Edit) để dọn dẹp format
        answer = await self._final_edit(query, answer)

        citations = [ref["match"] for ref in extract_article_references(answer)]
        return answer, num_corrections, citations

    async def _final_edit(self, query: str, raw_answer: str) -> str:
        prompt = FINAL_EDITOR_PROMPT.format(query=query, raw_answer=raw_answer)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.1)
        return response.text.strip()

    async def _detect_negative(self, query: str, answer: str) -> bool:
        if len(answer) < 30:
            return False
        prompt = NEGATIVE_DETECTION_PROMPT.format(query=query, answer=answer[:500])
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.0)
        return "USEFUL" in response.text.strip().upper()

    async def _self_correct_loop(
        self, query: str, context: str, answer: str, max_rounds: int = 2
    ) -> tuple[str, int]:
        current = answer
        for round_idx in range(max_rounds):
            prompt_template = SELF_CORRECT_1 if round_idx == 0 else SELF_CORRECT_2
            prompt = prompt_template.format(
                query=query, context=context, current_answer=current
            )
            messages = [Message(role="user", content=prompt)]
            response = await self.llm.generate(messages, temperature=0.2)

            corrected = response.text.strip()
            if corrected.upper().startswith("OK"):
                logger.info(f"Self-correction: accepted after round {round_idx + 1}")
                return current, round_idx + 1

            current = corrected
            logger.info(f"Self-correction: updated after round {round_idx + 1}")

        return current, max_rounds
