from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    article_id: str
    doc_title: str
    content: str
    score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""  # dense, sparse, hyde, query_expansion
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    tokens_input: int = 0
    tokens_output: int = 0
    model: str = ""


@dataclass
class QueryResult:
    query: str
    expanded_queries: list[str] = field(default_factory=list)
    query_type: str = "unknown"
    answer: str = ""
    corrected_answer: str = ""
    final_answer: str = ""
    relevant_docs: list[str] = field(default_factory=list)
    relevant_articles: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0
    num_correction_rounds: int = 0


class QueryType(str, Enum):
    YES_NO = "yes_no"
    FACTUAL = "factual"
    MULTI_ARTICLE = "multi_article"
    INTERPRETATION = "interpretation"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    CONDITION = "condition"
    SCENARIO = "scenario"


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    async def generate_stream(self, messages: list[Message], **kwargs) -> str:
        ...


class BaseEmbedding(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 500) -> list[RetrievedChunk]:
        ...


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 50) -> list[RetrievedChunk]:
        ...


class BaseQueryExpander(ABC):
    @abstractmethod
    async def expand(self, query: str, num_variations: int = 3) -> list[str]:
        ...


class BaseQueryClassifier(ABC):
    @abstractmethod
    async def classify(self, query: str) -> QueryType:
        ...


@dataclass
class ChunkStrategy:
    chunk_size: int
    overlap: int
    respect_boundaries: bool = True
    hierarchical: bool = False
