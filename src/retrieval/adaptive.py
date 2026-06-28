import logging

from src.core.base import BaseLLM, BaseQueryClassifier, Message, QueryType

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """Phân loại câu hỏi pháp luật sau đây vào một trong các loại:

- yes_no: Câu hỏi có/không, câu trả lời chỉ cần đúng/sai
- factual: Câu hỏi về sự kiện, quy định cụ thể (mức phạt, thời hạn, định nghĩa)
- condition: Câu hỏi về điều kiện, yêu cầu, tiêu chí để được hưởng quyền lợi hoặc thực hiện nghĩa vụ
- procedure: Câu hỏi về thủ tục, quy trình, hồ sơ, các bước thực hiện
- scenario: Câu hỏi tình huống giả định, đặt ra một hoàn cảnh cụ thể và hỏi về quyền/nghĩa vụ pháp lý
- multi_article: Câu hỏi cần tổng hợp từ nhiều điều luật hoặc nhiều văn bản khác nhau
- interpretation: Câu hỏi cần giải thích, diễn giải pháp luật
- comparison: Câu hỏi so sánh giữa các quy định

Câu hỏi: {query}

Chỉ trả về tên loại (ví dụ: factual), không kèm giải thích."""


ADAPTIVE_CONFIG = {
    QueryType.YES_NO: {
        "top_k_retrieval": 200, "top_k_rerank": 20, "top_k_final": 5,
        "num_query_variations": 1, "use_hyde": False,
        "dense_weight": 0.85, "sparse_weight": 0.15,
    },
    QueryType.FACTUAL: {
        "top_k_retrieval": 500, "top_k_rerank": 50, "top_k_final": 15,
        "num_query_variations": 2, "use_hyde": True,
        "dense_weight": 0.75, "sparse_weight": 0.25,
    },
    QueryType.CONDITION: {
        "top_k_retrieval": 600, "top_k_rerank": 60, "top_k_final": 20,
        "num_query_variations": 3, "use_hyde": True,
        "dense_weight": 0.75, "sparse_weight": 0.25,
    },
    QueryType.PROCEDURE: {
        "top_k_retrieval": 600, "top_k_rerank": 60, "top_k_final": 20,
        "num_query_variations": 2, "use_hyde": True,
        "dense_weight": 0.80, "sparse_weight": 0.20,
    },
    QueryType.SCENARIO: {
        "top_k_retrieval": 1000, "top_k_rerank": 80, "top_k_final": 25,
        "num_query_variations": 4, "use_hyde": True,
        "dense_weight": 0.70, "sparse_weight": 0.30,
    },
    QueryType.MULTI_ARTICLE: {
        "top_k_retrieval": 1000, "top_k_rerank": 80, "top_k_final": 25,
        "num_query_variations": 4, "use_hyde": True,
        "dense_weight": 0.70, "sparse_weight": 0.30,
    },
    QueryType.INTERPRETATION: {
        "top_k_retrieval": 800, "top_k_rerank": 70, "top_k_final": 20,
        "num_query_variations": 3, "use_hyde": True,
        "dense_weight": 0.90, "sparse_weight": 0.10,
    },
    QueryType.COMPARISON: {
        "top_k_retrieval": 1000, "top_k_rerank": 80, "top_k_final": 30,
        "num_query_variations": 4, "use_hyde": True,
        "dense_weight": 0.70, "sparse_weight": 0.30,
    },
}


class LLMQueryClassifier(BaseQueryClassifier):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def classify(self, query: str) -> QueryType:
        prompt = CLASSIFY_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.0)
        label = response.text.strip().lower()
        for qtype in QueryType:
            if qtype.value in label:
                logger.info(f"Query classified as: {qtype.value}")
                return qtype
        logger.warning(f"Could not classify query, defaulting to factual. Raw: {label}")
        return QueryType.FACTUAL


def get_adaptive_config(qtype: QueryType) -> dict:
    return ADAPTIVE_CONFIG.get(qtype, ADAPTIVE_CONFIG[QueryType.FACTUAL])
