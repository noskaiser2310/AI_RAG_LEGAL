import logging
import time

import numpy as np

from src.core.base import QueryResult, QueryType
from src.core.config import config
from src.generator.generator import Generator
from src.llm.hf_client import HFClient
from src.reranker.cross_encoder import CrossEncoderReranker
from src.retrieval.adaptive import LLMQueryClassifier, get_adaptive_config
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.indexing import DenseIndex, SparseIndex, build_indexes
from src.retrieval.multi_strategy import MultiStrategyRetriever
from src.retrieval.query_expansion import LLMQueryExpander
from src.retrieval.text_processor import (
    normalize_vietnamese_light,
    expand_legal_abbreviations,
    extract_article_references,
)

logger = logging.getLogger(__name__)

E5_INSTRUCT_PREFIX = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"

DECOMPOSE_PROMPT = """Phân tích câu hỏi pháp luật sau và tách thành các câu hỏi con độc lập.

Câu hỏi: {query}

Hướng dẫn phân tích:
1. Xác định hành vi/sự kiện pháp lý chính -> câu hỏi về quyền/nghĩa vụ
2. Xác định các điều kiện/tiêu chí cần đáp ứng -> câu hỏi về điều kiện
3. Xác định thủ tục/hồ sơ cần thực hiện -> câu hỏi về thủ tục
4. Xác định chế tài/bồi thường/xử lý -> câu hỏi về hậu quả pháp lý

Yêu cầu:
- Mỗi câu hỏi con phải độc lập, có thể tra cứu riêng trong văn bản pháp luật
- Bao gồm đầy đủ thuật ngữ pháp lý liên quan
- Nếu câu hỏi chỉ có một vấn đề, trả về chính câu hỏi đó
- Trả về mỗi câu hỏi trên một dòng, bắt đầu bằng "- "

Ví dụ:
Câu hỏi: "Phần mềm bị sao chép trái phép, cách tính tổn thất và hồ sơ khiếu nại?"
- Hành vi nào bị coi là xâm phạm quyền tác giả đối với phần mềm?
- Nguyên tắc và căn cứ xác định thiệt hại do xâm phạm quyền sở hữu trí tuệ là gì?
- Hồ sơ và trình tự yêu cầu xử lý vi phạm quyền tác giả gồm những gì?"""

RECOMPOSE_PROMPT = """Bạn là chuyên gia pháp lý tổng hợp câu trả lời.

Câu hỏi gốc: {original_query}

Các câu trả lời cho từng phần:
{sub_answers}

Hãy tổng hợp thành một câu trả lời hoàn chỉnh, mạch lạc, có trích dẫn điều luật đầy đủ.
Loại bỏ thông tin trùng lặp, sắp xếp hợp lý theo từng vấn đề."""


class LegalRAGPipeline:
    def __init__(
        self,
        docs: list[dict],
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        embedder=None,
        llm=None,
        reranker=None,
        use_hyde: bool = True,
        use_query_expansion: bool = True,
        use_adaptive: bool = True,
        use_decomposition: bool = True,
    ):
        self.docs = docs
        self.use_hyde = use_hyde
        self.use_query_expansion = use_query_expansion
        self.use_adaptive = use_adaptive
        self.use_decomposition = use_decomposition

        self.embedder = embedder
        self.llm = llm or HFClient()
        self.reranker = reranker or CrossEncoderReranker()
        self.dense_index = dense_index
        self.sparse_index = sparse_index

        self.retriever = MultiStrategyRetriever(
            docs, dense_index, sparse_index, self.embedder
        )
        self.generator = Generator(self.llm, self.retriever)
        self.query_expander = LLMQueryExpander(self.llm) if use_query_expansion else None
        self.query_classifier = LLMQueryClassifier(self.llm) if use_adaptive else None
        self.hyde = HyDEGenerator(self.llm, self.embedder) if use_hyde else None

    @classmethod
    async def create(
        cls,
        docs: list[dict],
        dense_path: str | None = None,
        sparse_path: str | None = None,
        embedder=None,
        llm=None,
        reranker=None,
        use_hyde: bool = True,
        use_query_expansion: bool = True,
        use_adaptive: bool = True,
        use_decomposition: bool = True,
    ):
        if embedder is None:
            from src.embedding.harrier_embedding import HarrierEmbedding
            embedder = HarrierEmbedding()
        if llm is None:
            llm = HFClient()
        if reranker is None:
            from src.reranker.cross_encoder import LLMReranker, TwoStageReranker
            ce = CrossEncoderReranker()
            llm_reranker = LLMReranker(llm)
            reranker = TwoStageReranker(ce, llm_reranker)
        logger.info("Building/loading indexes...")
        dense_idx, sparse_idx = await build_indexes(
            docs, embedder, dense_path, sparse_path
        )
        return cls(
            docs=docs,
            dense_index=dense_idx,
            sparse_index=sparse_idx,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            use_hyde=use_hyde,
            use_query_expansion=use_query_expansion,
            use_adaptive=use_adaptive,
            use_decomposition=use_decomposition,
        )

    async def _prepare_query(self, query: str, qtype: QueryType) -> str:
        norm = normalize_vietnamese_light(query)
        expanded = expand_legal_abbreviations(norm)
        if expanded != query:
            logger.info(f"Query prepared: {query[:60]} -> {expanded[:60]}")
        return expanded

    async def _embed_query(self, query: str) -> np.ndarray:
        prefixed = f"{E5_INSTRUCT_PREFIX}\nTruy vấn: {query}"
        emb_list = await self.embedder.embed([prefixed])
        return np.array(emb_list[0])

    async def _decompose_question(self, query: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(query=query)
        from src.core.base import Message
        response = await self.llm.generate(
            [Message(role="user", content=prompt)], temperature=0.1
        )
        sub_questions = [
            line.strip().lstrip("- ").strip()
            for line in response.text.strip().split("\n")
            if line.strip().startswith("-")
        ]
        sub_questions = [q for q in sub_questions if q and len(q) > 5]
        if len(sub_questions) <= 1:
            return [query]
        logger.info(f"Question decomposed: {query[:60]} -> {len(sub_questions)} sub-questions")
        return sub_questions

    async def _retrieve_for_query(
        self,
        query: str,
        top_k: int,
        seen_ids: set,
        all_chunks: list,
        q_emb: np.ndarray | None = None,
    ) -> None:
        if q_emb is None:
            q_emb = await self._embed_query(query)
        chunks = self.retriever.retrieve_sync(query, top_k, q_emb)
        for c in chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)

    async def _reconstruct_articles(self, chunks: list) -> list:
        article_map = {}
        no_article_chunks = []
        for c in chunks:
            if not c.article_id:
                no_article_chunks.append(c)
                continue
            key = (c.doc_title, c.article_id)
            if key not in article_map:
                article_map[key] = {
                    "doc_title": c.doc_title,
                    "article_id": c.article_id,
                    "sections": [],
                    "max_score": c.score,
                    "first_chunk": c,
                }
            article_map[key]["sections"].append(c.content)
            article_map[key]["max_score"] = max(article_map[key]["max_score"], c.score)

        reconstructed = []
        seen_articles = set()
        for c in chunks:
            if not c.article_id:
                continue
            key = (c.doc_title, c.article_id)
            if key not in seen_articles:
                seen_articles.add(key)
                info = article_map[key]
                merged = "\n".join(info["sections"])
                if len(merged) < 50:
                    merged = c.content
                from src.core.base import RetrievedChunk
                merged_chunk = RetrievedChunk(
                    chunk_id=f"merged_{c.doc_title}_{c.article_id}",
                    doc_id=c.doc_id,
                    article_id=c.article_id,
                    doc_title=c.doc_title,
                    content=merged,
                    score=info["max_score"],
                    retrieval_score=c.retrieval_score,
                    rerank_score=c.rerank_score,
                    source="reconstructed",
                    metadata=c.metadata,
                )
                reconstructed.append(merged_chunk)
        reconstructed.extend(no_article_chunks)
        reconstructed.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Reconstructed {len(reconstructed)} articles from {len(chunks)} chunks")
        return reconstructed

    async def _multi_hop_retrieve(
        self,
        chunks: list,
        original_query: str,
        seen_ids: set,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list:
        import re
        from src.core.base import RetrievedChunk

        ref_patterns = [
            r"(?:theo|tại|quy định tại|căn cứ)\s+(Điều\s+\d+[^.]*?)(?:\s+của|\s+Luật|\s+Nghị|\s+Thông|$)",
            r"(Nghị định\s+\d+/\d+/NĐ-\w+[^.]*?Điều\s+\d+)",
            r"(Thông tư\s+\d+/\d+/TT-\w+[^.]*?Điều\s+\d+)",
            r"(Luật\s+\d+/\d+/QH\d+[^.]*?Điều\s+\d+)",
        ]

        follow_up_queries = set()
        for c in chunks[:20]:
            for pattern in ref_patterns:
                for m in re.finditer(pattern, c.content[:2000]):
                    ref = m.group(1).strip()
                    if len(ref) > 5 and len(ref) < 200:
                        follow_up_queries.add(ref)

        if not follow_up_queries:
            return []

        logger.info(f"Multi-hop: {len(follow_up_queries)} follow-up queries from cross-references")

        new_chunks = []
        for fq in list(follow_up_queries)[:10]:
            q_emb = await self._embed_query(fq)
            fq_chunks = self.retriever.retrieve_sync(
                fq, top_k // 2, q_emb,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            for c in fq_chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    c.source = "multi_hop"
                    c.score *= 0.85
                    new_chunks.append(c)

        return new_chunks

    async def answer(
        self,
        query: str,
        use_self_correct: bool = True,
    ) -> QueryResult:
        t0 = time.time()
        result = QueryResult(query=query)

        qtype = QueryType.FACTUAL
        if self.query_classifier:
            qtype = await self.query_classifier.classify(query)
        result.query_type = qtype.value

        query = await self._prepare_query(query, qtype)

        adaptive_cfg = get_adaptive_config(qtype)
        top_k_retrieval = adaptive_cfg["top_k_retrieval"]
        top_k_rerank = adaptive_cfg["top_k_rerank"]
        num_variations = adaptive_cfg["num_query_variations"]
        use_hyde_q = adaptive_cfg["use_hyde"] and self.use_hyde

        logger.info(f"Query={query[:80]} type={qtype.value} top_k={top_k_retrieval}")

        sub_queries = [query]
        if self.use_decomposition and qtype in (QueryType.MULTI_ARTICLE, QueryType.COMPARISON, QueryType.INTERPRETATION, QueryType.SCENARIO):
            sub_queries = await self._decompose_question(query)
            result.expanded_queries = sub_queries[1:] if len(sub_queries) > 1 else []

        all_retrieval_queries = []
        for sq in sub_queries:
            all_retrieval_queries.append(sq)
            if self.query_expander and num_variations > 0:
                expanded = await self.query_expander.expand(sq, num_variations=num_variations)
                all_retrieval_queries.extend(expanded[1:])

        dense_weight = adaptive_cfg.get("dense_weight", 0.8)
        sparse_weight = adaptive_cfg.get("sparse_weight", 0.2)

        all_chunks = []
        seen_ids = set()

        for q in all_retrieval_queries:
            q_emb = await self._embed_query(q)
            chunks = self.retriever.retrieve_sync(
                q, top_k_retrieval, q_emb,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    all_chunks.append(c)

        if use_hyde_q and self.hyde:
            hypo_doc, hypo_emb = await self.hyde.generate_hypothetical(query)
            hyde_chunks = self.retriever.dense_search(np.array(hypo_emb), top_k_retrieval // 2)
            for c in hyde_chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    c.source = "hyde"
                    all_chunks.append(c)

        if qtype in (QueryType.MULTI_ARTICLE, QueryType.SCENARIO, QueryType.COMPARISON):
            hop_chunks = await self._multi_hop_retrieve(
                all_chunks, query, seen_ids, top_k_retrieval,
                dense_weight, sparse_weight,
            )
            all_chunks.extend(hop_chunks)

        t1 = time.time()
        result.retrieval_time = t1 - t0

        all_chunks = await self.reranker.rerank(query, all_chunks, top_k_rerank)

        all_chunks = await self._reconstruct_articles(all_chunks)

        t2 = time.time()
        result.rerank_time = t2 - t1

        if len(sub_queries) > 1:
            sub_answers = []
            for i, sq in enumerate(sub_queries):
                sq_chunks = [c for c in all_chunks if c.score > config.SCORE_THRESHOLD * 0.9]
                ans, _, _ = await self.generator.generate(
                    sq, chunks=sq_chunks, use_self_correct=False
                )
                sub_answers.append(f"Câu hỏi {i+1}: {sq}\nTrả lời: {ans}")

            recompose_prompt = RECOMPOSE_PROMPT.format(
                original_query=query,
                sub_answers="\n\n".join(sub_answers),
            )
            from src.core.base import Message
            response = await self.llm.generate(
                [Message(role="user", content=recompose_prompt)], temperature=0.1
            )
            final_answer = response.text
            num_corrections = 0
        else:
            final_answer, num_corrections, citations = await self.generator.generate(
                query,
                chunks=all_chunks,
                use_self_correct=use_self_correct,
                score_threshold=config.SCORE_THRESHOLD,
            )

        t3 = time.time()
        result.generation_time = t3 - t2
        result.answer = final_answer
        result.final_answer = final_answer
        result.num_correction_rounds = num_corrections
        refs = extract_article_references(final_answer)
        seen_cites = set()
        result.citations = []
        for ref in refs:
            match_str = ref["match"]
            if match_str not in seen_cites:
                seen_cites.add(match_str)
                result.citations.append(match_str)

        # Dynamic filtering of relevant chunks based on citations and scores
        used_chunks = []
        max_score = all_chunks[0].score if all_chunks else 0
        for c in all_chunks[:config.FINAL_TOP_K]:
            is_cited = False
            if c.article_id and (f"Điều {c.article_id}" in final_answer or f"điều {c.article_id}" in final_answer.lower()):
                is_cited = True
            elif c.doc_title and c.doc_title in final_answer:
                is_cited = True
                
            if is_cited or c.score >= max_score * 0.98:
                used_chunks.append(c)
                
        if not used_chunks and all_chunks:
            used_chunks = [all_chunks[0]]

        import re
        def clean_title(title: str) -> str:
            if not title: return ""
            return re.sub(r'\s*-\s*Điều\s+\w+.*$', '', title).strip()

        def format_article(article_id: str) -> str:
            if not article_id: return ""
            if "điều" not in article_id.lower():
                return f"Điều {article_id}"
            return article_id

        relevant_docs = list(set(f"{c.metadata.get('so_ky_hieu', c.doc_id)}|{clean_title(c.doc_title)}" for c in used_chunks if c.doc_title))
        relevant_articles = list(set(f"{c.metadata.get('so_ky_hieu', c.doc_id)}|{clean_title(c.doc_title)}|{format_article(c.article_id)}" for c in used_chunks if c.article_id))
        
        result.relevant_docs = relevant_docs
        result.relevant_articles = relevant_articles
        result.chunks = used_chunks
        result.confidence = max(c.score for c in all_chunks) if all_chunks else 0.0

        total = t3 - t0
        logger.info(
            f"Done: retrieval={result.retrieval_time:.1f}s "
            f"rerank={result.rerank_time:.1f}s "
            f"generation={result.generation_time:.1f}s "
            f"total={total:.1f}s "
            f"corrections={num_corrections} "
            f"chunks={len(all_chunks)} "
            f"sub_queries={len(sub_queries)}"
        )
        return result

    async def answer_agentic(
        self,
        query: str,
        use_self_correct: bool = True,
    ) -> QueryResult:
        from src.pipeline.agent import LegalAgent

        agent = LegalAgent(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker,
            embedder=self.embedder,
            docs=self.docs,
        )
        return await agent.answer(
            query,
            max_iterations=config.AGENTIC_MAX_ITERATIONS,
            use_self_correct=use_self_correct,
        )
