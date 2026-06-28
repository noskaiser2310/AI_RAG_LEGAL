from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "LegalRAG"

    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    EMBEDDING_MODEL: str = "mainguyen9/vietlegal-harrier-0.6b"
    EMBEDDING_DIM: int = 1024
    DEVICE: str = "cpu"

    RERANKER_MODEL: str = "AITeamVN/Vietnamese_Reranker"
    RERANKER_TOP_K: int = 30

    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    RETRIEVAL_TOP_K: int = 500
    RERANK_TOP_K: int = 50
    FINAL_TOP_K: int = 20
    SCORE_THRESHOLD: float = 0.7

    MAX_TOKENS: int = 8192
    TEMPERATURE: float = 0.1

    DATA_DIR: str = "data"
    INDEX_DIR: str = "data/indexes"
    CORPUS_FILE: str = "corpus.jsonl"

    MAX_CORPUS_CHUNKS: int = 0

    USE_VN_SEGMENTATION: bool = True
    USE_TIERED_CHUNKING: bool = False

    EVAL_TOP_K_RETRIEVAL: int = 500
    EVAL_TOP_K_RERANK: int = 50
    EVAL_MAX_QUERIES: int = 200

    AGENTIC_MAX_ITERATIONS: int = 2
    AGENTIC_TOP_K: int = 30

    TUNE_DENSE_WEIGHTS: str = "0.7,0.75,0.8,0.85,0.9"
    TUNE_SCORE_THRESHOLDS: str = "0.85,0.9,0.95"
    TUNE_BM25_K1_VALUES: str = "1.2,1.5,1.8,2.0"
    TUNE_GAP_RATIOS: str = "0.85,0.9,0.95"
    TUNE_MAX_COMBOS: int = 30


config = Config()
