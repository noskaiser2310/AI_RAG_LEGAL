import { useEffect, useRef, useState } from "react";

const SUGGESTIONS = [
  "Doanh nghiệp bị sao chép phần mềm trái phép thì xử lý thế nào?",
  "Thủ tục đăng ký thành lập công ty TNHH một thành viên?",
  "Người lao động nghỉ việc ngang có bị phạt không?",
  "Điều kiện để được hưởng bảo hiểm thất nghiệp?",
  "Thuế thu nhập doanh nghiệp áp dụng cho startup?",
];

export default function Composer({
  onSend,
  busy,
}: {
  onSend: (text: string) => void;
  busy: boolean;
}) {
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={taRef}
          rows={1}
          value={text}
          placeholder="Nhập câu hỏi pháp lý của bạn…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          className={`send-btn${busy ? " loading" : ""}`}
          onClick={submit}
          disabled={busy || !text.trim()}
          title="Gửi câu hỏi"
        >
          {busy ? "…" : "→"}
        </button>
      </div>
      <div className="composer-hint">
        LegalRAG trả lời dựa trên 1.064.000+ điều luật Việt Nam · Nhấn Enter để gửi, Shift+Enter xuống dòng
      </div>
      {!busy && (
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="suggestion" onClick={() => onSend(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
