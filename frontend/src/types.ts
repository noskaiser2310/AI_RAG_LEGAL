export interface RetrievedChunk {
  chunk_id: string;
  doc_id: string;
  article_id: string;
  doc_title: string;
  content: string;
  score: number;
  retrieval_score: number;
  rerank_score: number;
  source: string;
  metadata: Record<string, unknown>;
}

export interface ChatResponse {
  query: string;
  query_type: string;
  answer: string;
  citations: string[];
  relevant_docs: string[];
  relevant_articles: string[];
  expanded_queries: string[];
  confidence: number;
  retrieval_time: number;
  rerank_time: number;
  generation_time: number;
  total_time: number;
  num_correction_rounds: number;
  chunks: RetrievedChunk[];
}

export interface HealthInfo {
  status: "loading" | "ready" | "error";
  app: string;
  corpus_docs: number | null;
  dense_index_vectors: number | null;
  sparse_index_docs: number | null;
  llm_backend: string;
  llm_model: string;
  device: string;
  embedding_model: string;
  reranker_model: string;
  detail?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "thinking" | "streaming" | "done" | "error";
  error?: string;
  result?: ChatResponse;
}
