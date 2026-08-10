import asyncio
import json
import logging
import time
import re

import numpy as np

from src.core.base import BaseLLM, Message, QueryResult, RetrievedChunk
from src.core.config import config
from src.generator.generator import Generator, format_context
from src.retrieval.text_processor import extract_article_references, extract_doc_code

logger = logging.getLogger(__name__)

LAW_ANALYSIS_PROMPT = """Bạn là luật sư Việt Nam giàu kinh nghiệm. Phân tích câu hỏi pháp lý sau và xác định các nguồn luật liên quan.

Câu hỏi: {query}

Hãy trả lời bằng JSON với cấu trúc sau:
{{
  "domains": ["lĩnh vực pháp luật liên quan, ví dụ: lao động, thuế, doanh nghiệp"],
  "laws": ["tên đầy đủ các văn bản luật có thể áp dụng, ví dụ: Luật Lao động 2019, Bộ luật Dân sự 2015, Nghị định 12/2022/NĐ-CP"],
  "key_articles": ["các điều khoản có thể liên quan, ví dụ: Điều 36 Luật Lao động về đơn phương chấm dứt hợp đồng"],
  "search_queries": ["2-3 truy vấn ngắn gọn bằng tiếng Việt để tìm điều luật liên quan trong cơ sở dữ liệu"]
}}

Lưu ý:
- Liệt kê TẤT CẢ văn bản luật có thể liên quan (Luật, Nghị định, Thông tư)
- Bao gồm cả văn bản hướng dẫn chi tiết (Nghị định, Thông tư) không chỉ Luật
- search_queries phải là các câu hỏi/truy vấn cụ thể có thể dùng để tra cứu
- Chỉ trả về JSON, không kèm giải thích khác."""

ASSESS_PROMPT = """Bạn là luật sư đánh giá độ phủ thông tin pháp lý.

Câu hỏi cần trả lời: {query}

Các điều luật đã tìm được:
{context_summary}

Hãy đánh giá và trả lời bằng JSON:
{{
  "is_complete": true hoặc false,
  "confidence": 0.0 đến 1.0,
  "gaps": "mô tả ngắn những thông tin còn thiếu (nếu có)",
  "follow_up_queries": ["2-3 truy vấn cụ thể để tìm thêm thông tin còn thiếu, bao gồm tên văn bản và điều khoản"]
}}

Hướng dẫn:
- is_complete = true NẾU đã có đủ điều luật để trả lời câu hỏi
- is_complete = false NẾU còn thiếu văn bản, điều khoản quan trọng
- follow_up_queries chỉ cần khi is_complete = false
- Mỗi follow_up_query nên cụ thể: "Điều X Nghị định Y về vấn đề Z"
- Chỉ trả về JSON, không kèm giải thích khác."""

HYDE_PROMPT = """Bạn là một chuyên gia pháp lý tại Việt Nam. 
Hãy viết một đoạn văn ngắn (khoảng 3-4 câu) bằng văn phong pháp luật trang trọng để trả lời trực tiếp cho câu hỏi sau. 
Không cần giải thích dài dòng, hãy viết như thể bạn đang trích dẫn hoặc tóm tắt từ một văn bản luật thực sự.

Câu hỏi: {query}

Câu trả lời giả định:"""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
    return {}


_LAW_ABBREVS = {
    "dnnvv": "doanh nghiệp nhỏ và vừa",
    "shtt": "sở hữu trí tuệ",
    "bllđ": "bộ luật lao động",
    "ld": "lao động",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "tncn": "thu nhập cá nhân",
    "gtgt": "giá trị gia tăng",
    "xd": "xây dựng",
    "bvmt": "bảo vệ môi trường",
    "ntd": "người tiêu dùng",
}


class LawNameIndex:
    def __init__(self, docs: list[dict]):
        self.code_to_titles: dict[str, list[str]] = {}
        self.all_titles: list[str] = []
        self.name_to_title: dict[str, str] = {}
        seen_titles = set()
        for doc in docs:
            title = doc.get("title", "")
            if not title:
                continue
            title = re.sub(r"\s*-\s*Điều\s+\d+\s*$", "", title)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            self.all_titles.append(title)
            code = doc.get("so_ky_hieu", "") or extract_doc_code(title)
            if code:
                self.code_to_titles.setdefault(code, []).append(title)

            name = re.sub(r"\d+/\d{4}/\S+\s*", "", title).strip()
            name = re.sub(r"^(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Chỉ thị)\s+", "", name).strip()
            if len(name) > 3:
                self.name_to_title[name.lower()] = title
                expanded = self._expand_abbrevs(name.lower())
                if expanded != name.lower():
                    self.name_to_title[expanded] = title

    @staticmethod
    def _expand_abbrevs(text: str) -> str:
        for abbr, full in _LAW_ABBREVS.items():
            text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text)
        return text

    def match(self, law_name: str) -> str | None:
        if not law_name or len(law_name) < 3:
            return None
        code = extract_doc_code(law_name)
        if code:
            matches = self.code_to_titles.get(code, [])
            if matches:
                return max(matches, key=len)

        law_lower = law_name.lower().strip()
        law_type = self._law_type(law_lower)
        year_m = re.search(r"\b(19|20)\d{2}\b", law_lower)
        law_year = year_m.group(0) if year_m else None
        core = self._core_name(law_lower)
        core_words = [w for w in core.split() if len(w) > 2]

        # 1) Type-aware scored match (dung loai van ban + uu tien dung nam)
        if core_words:
            best_match = None
            best_score = 0.0
            for title in self.all_titles:
                if len(title) < 10:
                    continue
                title_lower = title.lower()
                t_type = self._law_type(title_lower)
                if law_type:
                    if not self._types_compatible(law_type, t_type):
                        continue
                t_year_m = re.search(r"\b(19|20)\d{2}\b", title_lower)
                t_year = t_year_m.group(0) if t_year_m else None
                t_core = self._core_name(title_lower)
                t_words = set(w for w in t_core.split() if len(w) > 2)
                if not t_words:
                    continue
                hits = sum(1 for w in core_words if w in t_words)
                ratio = hits / len(core_words)
                if ratio < 0.6 or hits < 2:
                    continue
                score = ratio
                if law_year and t_year == law_year:
                    score += 0.2
                elif law_year and t_year:
                    score -= 0.1
                if score > best_score:
                    best_score = score
                    best_match = title
            if best_match:
                return best_match

        # 2) Fallback: ten da chuan hoa (van tuan thu loai van ban)
        law_core = re.sub(r"^(bộ luật|luật|nghị định|thông tư|quyết định|chỉ thị)\s+", "", law_lower).strip()
        law_core = re.sub(r"\s*\d{4}\s*$", "", law_core).strip()
        if len(law_core) > 3:
            for name_key, title in self.name_to_title.items():
                t_type = self._law_type(title.lower())
                if law_type and not self._types_compatible(law_type, t_type):
                    continue
                if law_core in name_key or name_key in law_core:
                    return title

        return None

    @staticmethod
    def _law_type(text: str) -> str | None:
        m = re.match(
            r"^\s*(bộ luật|thông tư liên tịch|nghị định|thông tư|quyết định|chỉ thị|nghị quyết|pháp lệnh|luật)\b",
            text,
        )
        return m.group(1) if m else None

    @staticmethod
    def _types_compatible(q_type: str, t_type: str | None) -> bool:
        if not t_type:
            return True
        law_like = {"luật", "bộ luật"}
        if q_type in law_like and t_type in law_like:
            return True
        return q_type == t_type

    @staticmethod
    def _core_name(text: str) -> str:
        t = re.sub(r"^\s*(bộ luật|luật|nghị định|thông tư liên tịch|thông tư|quyết định|chỉ thị|nghị quyết|pháp lệnh)\s+", "", text)
        t = re.sub(r"\b\d{4}\b", "", t)
        t = re.sub(r"\d+/\d+/[^\s]+", "", t)
        t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
        return re.sub(r"\s+", " ", t).strip()


class LegalAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever,
        reranker,
        embedder,
        docs: list[dict],
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder
        self.docs = docs
        self.generator = Generator(llm, retriever)
        self.law_index = LawNameIndex(docs)

    async def answer(
        self,
        query: str,
        max_iterations: int = 0,
        use_self_correct: bool = True,
    ) -> QueryResult:
        if max_iterations <= 0:
            max_iterations = config.AGENTIC_MAX_ITERATIONS
        t0 = time.time()
        result = QueryResult(query=query)

        logger.info(f"[Agent] Analyzing: {query[:80]}")
        law_analysis = await self._analyze_laws(query)
        result.query_type = "agentic"

        domains = law_analysis.get("domains", [])
        laws = law_analysis.get("laws", [])
        search_queries = law_analysis.get("search_queries", [])
        logger.info(
            f"[Agent] Domains: {domains[:3]}, "
            f"Laws: {len(laws)}, "
            f"Search queries: {len(search_queries)}"
        )

        is_situational = not bool(re.search(r'(điều kiện|tiêu chí|khái niệm|quy định|là gì|thế nào)', query.lower()))
        
        hyde_doc = ""
        if is_situational:
            logger.info(f"[Agent] Generating HyDE document...")
            prompt = HYDE_PROMPT.format(query=query)
            hyde_response = await self.llm.generate([Message(role="user", content=prompt)], temperature=0.7)
            hyde_doc = hyde_response.text.strip()
            logger.info(f"[Agent] HyDE document generated ({len(hyde_doc)} chars)")
        else:
            logger.info(f"[Agent] Skipping HyDE for definition/factual query.")

        law_queries = []
        for law in laws:
            matched = self.law_index.match(law)
            if matched:
                law_queries.append(matched)
                logger.info(f"[Agent] Law matched: {law[:40]} -> {matched[:60]}")
            else:
                logger.info(f"[Agent] Law unmatched: {law[:40]}")

        retrieval_queries = [query, hyde_doc] + search_queries

        all_chunks: list[RetrievedChunk] = []
        seen_ids: set = set()

        logger.info(f"[Agent] Initial retrieval...")
        retrieval_top_k = config.RETRIEVAL_TOP_K * 4 if law_queries else config.RETRIEVAL_TOP_K
        
        initial_chunks = await self._retrieve_with_queries(
            queries=retrieval_queries,
            seen_ids=seen_ids,
            top_k=retrieval_top_k,
            predicted_laws=law_queries
        )
        
        if law_queries:
            filtered = [c for c in initial_chunks if c.doc_title and c.doc_title in law_queries]
            if len(filtered) >= 10:
                initial_chunks = filtered
                logger.info(f"[Agent] Hard-filtered to {len(initial_chunks)} chunks matching laws.")
            else:
                logger.info(f"[Agent] Hard-filtering yielded too few results ({len(filtered)}). Falling back to soft-boosting.")
                
        all_chunks.extend(initial_chunks[:config.RETRIEVAL_TOP_K * 2])
        logger.info(f"[Agent] Initial: {len(all_chunks)} chunks")

        t1 = time.time()
        result.retrieval_time = t1 - t0

        top_k_rerank = config.RERANK_TOP_K
        iteration = 0
        for iteration in range(max_iterations):
            reranked = await self.reranker.rerank(query, all_chunks, top_k_rerank)

            assessment = await self._assess_coverage(query, reranked)
            is_complete = assessment.get("is_complete", True)
            confidence = assessment.get("confidence", 0.5)
            gaps = assessment.get("gaps", "")
            follow_up = assessment.get("follow_up_queries", [])

            logger.info(
                f"[Agent] Iteration {iteration + 1}: "
                f"complete={is_complete}, confidence={confidence:.2f}, "
                f"gaps={gaps[:80] if gaps else 'none'}, "
                f"follow_up={len(follow_up)} queries"
            )

            if is_complete or not follow_up:
                break

            new_chunks = await self._retrieve_with_queries(
                queries=follow_up, 
                seen_ids=seen_ids, 
                top_k=config.RETRIEVAL_TOP_K // 2,
                predicted_laws=law_queries
            )
            if new_chunks:
                for c in new_chunks:
                    c.source = f"agent_iter_{iteration + 1}"
                all_chunks.extend(new_chunks)
                logger.info(
                    f"[Agent] After iteration {iteration + 1}: "
                    f"+{len(new_chunks)} chunks, total={len(all_chunks)}"
                )
            else:
                logger.info(f"[Agent] No new chunks in iteration {iteration + 1}, stopping")
                break

        t_followup = time.time()
        result.retrieval_time = t_followup - t0

        final_chunks = await self.reranker.rerank(query, all_chunks, config.RERANK_TOP_K)

        final_chunks = await self._reconstruct_articles(final_chunks)

        t2 = time.time()
        result.rerank_time = t2 - t_followup

        top_k_final = config.AGENTIC_TOP_K
        final_answer, num_corrections, citations = await self.generator.generate(
            query,
            chunks=final_chunks[:top_k_final * 2],
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
        max_score = final_chunks[0].score if final_chunks else 0
        for c in final_chunks[:config.FINAL_TOP_K]:
            is_cited = False
            if c.article_id:
                art_clean = c.article_id.lower().replace("điều", "").strip()
                if f"điều {art_clean}" in final_answer.lower():
                    is_cited = True
            elif c.doc_title and c.doc_title in final_answer:
                is_cited = True
                
            if is_cited or c.score >= max_score * 0.98:
                used_chunks.append(c)
                
        if not used_chunks and final_chunks:
            used_chunks = [final_chunks[0]]

        def clean_title(title: str) -> str:
            if not title: return ""
            return re.sub(r'\s*-\s*Điều\s+\w+.*$', '', title).strip()

        def format_article(article_id: str) -> str:
            if not article_id: return ""
            if "điều" not in article_id.lower():
                return f"Điều {article_id}"
            return article_id

        result.relevant_docs = list(set(f"{c.metadata.get('so_ky_hieu', c.doc_id)}|{clean_title(c.doc_title)}" for c in used_chunks if c.doc_title))
        result.relevant_articles = list(set(f"{c.metadata.get('so_ky_hieu', c.doc_id)}|{clean_title(c.doc_title)}|{format_article(c.article_id)}" for c in used_chunks if c.article_id))
        result.chunks = used_chunks[:config.FINAL_TOP_K]
        result.confidence = max(c.score for c in final_chunks) if final_chunks else 0.0
        result.expanded_queries = search_queries

        total = t3 - t0
        logger.info(
            f"[Agent] Done: retrieval={result.retrieval_time:.1f}s "
            f"rerank={result.rerank_time:.1f}s "
            f"generation={result.generation_time:.1f}s "
            f"total={total:.1f}s "
            f"iterations={iteration + 1} "
            f"chunks={len(final_chunks)}"
        )
        return result

    async def _analyze_laws(self, query: str) -> dict:
        prompt = LAW_ANALYSIS_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.1)
        parsed = _parse_json_response(response.text)
        if not parsed:
            return {"domains": [], "laws": [], "key_articles": [], "search_queries": []}
        return parsed

    async def _assess_coverage(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> dict:
        context_summary = self._summarize_chunks(chunks)
        prompt = ASSESS_PROMPT.format(query=query, context_summary=context_summary)
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages, temperature=0.0)
        parsed = _parse_json_response(response.text)
        if not parsed:
            return {"is_complete": True, "confidence": 0.5, "gaps": "", "follow_up_queries": []}
        return parsed

    async def _retrieve_with_queries(
        self,
        queries: list[str],
        seen_ids: set,
        top_k: int = 500,
        predicted_laws: list[str] = None
    ) -> list[RetrievedChunk]:
        predicted_laws = predicted_laws or []
        prefix = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"

        async def _embed_and_retrieve(q: str):
            prefixed = f"{prefix}\nTruy vấn: {q}"
            emb_list = await self.embedder.embed([prefixed])
            q_emb = np.array(emb_list[0])
            
            # Dynamic Hybrid Weighting
            q_lower = q.lower()
            if re.search(r'(điều\s+\d+|luật\s+|nghị định\s+|thông tư\s+)', q_lower):
                dense_w, sparse_w = 0.3, 0.7  # Câu hỏi chứa keyword cứng -> Ưu tiên BM25
            else:
                dense_w, sparse_w = 0.7, 0.3  # Câu hỏi tình huống -> Ưu tiên Dense
                
            return self.retriever.retrieve_sync(
                query=q, 
                top_k=top_k, 
                query_emb=q_emb,
                dense_weight=dense_w,
                sparse_weight=sparse_w
            )

        tasks = [_embed_and_retrieve(q) for q in queries if q.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Retrieval failed: {result}")
                continue
            for c in result:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    # Metadata Score Boosting
                    if c.doc_title and c.doc_title in predicted_laws:
                        c.score *= 1.2
                    new_chunks.append(c)
        return new_chunks

    def _summarize_chunks(self, chunks: list[RetrievedChunk], max_chunks: int = 20) -> str:
        parts = []
        seen = set()
        for i, c in enumerate(chunks[:max_chunks]):
            key = f"{c.doc_title}|{c.article_id}"
            if key in seen:
                continue
            seen.add(key)
            title = c.doc_title or "Văn bản"
            article = f" - Điều {c.article_id}" if c.article_id else ""
            content_preview = c.content[:300].replace("\n", " ")
            parts.append(f"[{i+1}] {title}{article}: {content_preview}")
        return "\n".join(parts)

    async def _reconstruct_articles(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        article_map = {}
        no_article_chunks = []
        for c in chunks:
            if not c.article_id:
                no_article_chunks.append(c)
                continue
            key = (c.doc_title, c.article_id)
            if key not in article_map:
                article_map[key] = {
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
        logger.info(f"[Agent] Reconstructed {len(reconstructed)} articles from {len(chunks)} chunks")
        return reconstructed
