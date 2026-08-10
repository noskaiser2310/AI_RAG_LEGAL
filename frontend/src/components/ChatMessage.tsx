import { useState } from "react";
import type { ChatMessage } from "../types";
import Markdown from "./Markdown";

function fmtTime(s: number): string {
  if (!s) return "0s";
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  return `${s.toFixed(1)}s`;
}

function fmtType(t: string): string {
  const map: Record<string, string> = {
    factual: "Tra cứu",
    multi_article: "Nhiều điều",
    interpretation: "Giải thích",
    procedure: "Thủ tục",
    comparison: "So sánh",
    condition: "Điều kiện",
    scenario: "Tình huống",
    yes_no: "Hỏi đáp",
    agentic: "Phân tích",
    unknown: "Tổng hợp",
    retrieve: "Tra cứu",
  };
  return map[t] || t;
}

function Thinking() {
  return (
    <div className="thinking">
      <span>Đang tra cứu &amp; tổng hợp điều luật</span>
      <span className="dots">
        <i />
        <i />
        <i />
      </span>
    </div>
  );
}

function SourceItem({
  chunk,
  rank,
  idx,
}: {
  chunk: import("../types").RetrievedChunk;
  rank: number;
  idx: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const art = chunk.article_id ? `Điều ${chunk.article_id}` : "";
  const so = (chunk.metadata?.so_ky_hieu as string) || "";
  return (
    <div className="source-item" key={`${chunk.chunk_id}_${idx}`}>
      <div className="src-head">
        <span className="src-rank">{rank}</span>
        <span className="src-title">
          {chunk.doc_title}
          {art ? ` — ${art}` : ""}
        </span>
        <span className="src-score">{chunk.score.toFixed(3)}</span>
      </div>
      {so && <span className="src-tag">{so}</span>}
      {chunk.source && <span className="src-tag">{chunk.source}</span>}
      <div className={`src-content${expanded ? " expanded" : ""}`} onClick={() => setExpanded(!expanded)}>
        {chunk.content.slice(0, expanded ? undefined : 400)}
      </div>
      <button className="src-expand" onClick={() => setExpanded(!expanded)}>
        {expanded ? "Thu gọn" : "Xem đầy đủ"}
      </button>
    </div>
  );
}

export default function ChatMessageView({
  msg,
}: {
  msg: ChatMessage;
}) {
  if (msg.role === "user") {
    return (
      <div className="msg user">
        <div className="msg-body">
          <div className="msg-label">Câu hỏi</div>
          <div className="bubble">{msg.content}</div>
        </div>
        <div className="msg-avatar">H</div>
      </div>
    );
  }

  const r = msg.result;

  return (
    <div className={`msg assistant${msg.status === "error" ? " error" : ""}`}>
      <div className="msg-avatar">L</div>
      <div className="msg-body">
        <div className="msg-label">Trợ lý pháp lý</div>
        <div className="card">
          {msg.status === "thinking" && !r && (
            <div className="card-body">
              <Thinking />
            </div>
          )}

          {msg.status === "error" && (
            <div className="card-body">
              <div className="error-box">{msg.error || "Đã xảy ra lỗi khi xử lý câu hỏi."}</div>
            </div>
          )}

          {r && (
            <>
              <div className="card-head">
                <span className="badge accent">{fmtType(r.query_type)}</span>
                {r.num_correction_rounds > 0 && (
                  <span className="badge">{r.num_correction_rounds} lần kiểm tra</span>
                )}
                <span className="badge">{r.chunks.length} nguồn</span>
                <div className="confidence-bar" title="Mức độ tin cậy">
                  <div className="track">
                    <div className="fill" style={{ width: `${Math.round(r.confidence * 100)}%` }} />
                  </div>
                  <span>{Math.round(r.confidence * 100)}%</span>
                </div>
              </div>
              <div className="card-body">
                <Markdown text={r.answer} />
              </div>
              <div className="metrics">
                <span className="metric">
                  Tra cứu <b>{fmtTime(r.retrieval_time)}</b>
                </span>
                <span className="metric">
                  Rerank <b>{fmtTime(r.rerank_time)}</b>
                </span>
                <span className="metric">
                  Sinh câu trả lời <b>{fmtTime(r.generation_time)}</b>
                </span>
                <span className="metric">
                  Tổng <b>{fmtTime(r.total_time)}</b>
                </span>
              </div>
              {r.chunks.length > 0 && <Sources chunks={r.chunks} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Sources({ chunks }: { chunks: import("../types").RetrievedChunk[] }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        className={`sources-toggle${open ? " open" : ""}`}
        onClick={() => setOpen(!open)}
      >
        <span>Điều luật trích dẫn · {chunks.length}</span>
        <span className="chev">▼</span>
      </button>
      {open && (
        <div className="sources">
          {chunks.map((c, i) => (
            <SourceItem key={c.chunk_id} chunk={c} rank={i + 1} idx={i} />
          ))}
        </div>
      )}
    </>
  );
}
