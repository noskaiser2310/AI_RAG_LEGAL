import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, streamChat } from "./api";
import ChatMessageView from "./components/ChatMessage";
import Composer from "./components/Composer";
import type { ChatMessage, ChatResponse, HealthInfo } from "./types";

interface Conversation {
  id: string;
  title: string;
  time: number;
  messages: ChatMessage[];
}

const LS_KEY = "legalrag.conversations.v1";
const LS_THEME = "legalrag.theme";

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function loadHistory(): Conversation[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadHistory);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (localStorage.getItem(LS_THEME) as "light" | "dark") || "light",
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(LS_THEME, theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(conversations.slice(0, 30)));
  }, [conversations]);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const active = conversations.find((c) => c.id === activeId) || null;

  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: Partial<ChatMessage>) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id !== convId
            ? c
            : {
                ...c,
                messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m)),
              },
        ),
      );
    },
    [],
  );

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  }, []);

  const newConversation = useCallback(() => {
    const id = uid();
    setConversations((prev) => [
      { id, title: "Cuộc trò chuyện mới", time: Date.now(), messages: [] },
      ...prev,
    ]);
    setActiveId(id);
    setSidebarOpen(false);
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      let convId = activeId;
      if (!convId) {
        convId = uid();
        setConversations((prev) => [
          { id: convId!, title: text.slice(0, 60), time: Date.now(), messages: [] },
          ...prev,
        ]);
        setActiveId(convId);
      }

      const userMsg: ChatMessage = { id: uid(), role: "user", content: text, status: "done" };
      const asstMsg: ChatMessage = { id: uid(), role: "assistant", content: "", status: "thinking" };

      setConversations((prev) =>
        prev.map((c) =>
          c.id !== convId
            ? c
            : {
                ...c,
                title: c.title === "Cuộc trò chuyện mới" ? text.slice(0, 60) : c.title,
                time: Date.now(),
                messages: [...c.messages, userMsg, asstMsg],
              },
        ),
      );

      scrollToBottom();
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setBusy(true);

      try {
        await streamChat(
          text,
          "agentic",
          true,
          (result: ChatResponse) => {
            patchMessage(convId!, asstMsg.id, {
              status: "done",
              content: result.answer,
              result,
            });
            scrollToBottom();
          },
          (err) => {
            patchMessage(convId!, asstMsg.id, { status: "error", error: err });
          },
          ac.signal,
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        patchMessage(convId!, asstMsg.id, {
          status: "error",
          error: "Không thể kết nối tới máy chủ. Hãy chạy backend: uvicorn api.main:app --port 8000",
        });
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [activeId, patchMessage, scrollToBottom],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) setActiveId(null);
    },
    [activeId],
  );

  const statusLabel = health
    ? health.status === "ready"
      ? `${health.llm_model} · ${(health.corpus_docs ?? 0).toLocaleString("vi-VN")} văn bản`
      : health.status === "loading"
        ? "Đang khởi tạo…"
        : "Lỗi khởi tạo"
    : "Đang kết nối…";

  const statusClass = !health ? "loading" : health.status;

  return (
    <div className="app">
      <aside className={`sidebar${sidebarOpen ? " open" : ""}`}>
        <div className="brand">
          <div className="brand-seal">L</div>
          <div>
            <div className="brand-name">LegalRAG</div>
            <div className="brand-sub">Trợ lý pháp lý Việt Nam</div>
          </div>
        </div>

        <button className="new-chat" onClick={newConversation}>
          <span>✚</span> Câu hỏi mới
        </button>

        <div className="sidebar-section">Lịch sử tra cứu</div>
        <div className="history">
          {conversations.length === 0 && (
            <div style={{ padding: "12px", fontSize: 12.5, color: "rgba(232,226,212,0.45)", lineHeight: 1.6 }}>
              Chưa có cuộc tra cứu nào. Đặt câu hỏi pháp lý để bắt đầu.
            </div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`history-item${c.id === activeId ? " active" : ""}`}
              onClick={() => {
                setActiveId(c.id);
                setSidebarOpen(false);
              }}
            >
              <span className="item-title">{c.title}</span>
              <span className="item-time">
                {new Date(c.time).toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" })}
              </span>
              <button
                className="delete"
                title="Xóa"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(c.id);
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <span className={`status-dot ${statusClass}`} />
          <span className="detail">{statusLabel}</span>
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <div style={{ display: "flex", alignItems: "center" }}>
            <button className="menu-btn" onClick={() => setSidebarOpen(true)}>
              ☰
            </button>
            <div className="title">
              Tra cứu pháp luật
              {health?.status === "ready" && (
                <span className="doc-count">
                  {health.sparse_index_docs?.toLocaleString("vi-VN")} điều khoản trong cơ sở dữ liệu
                </span>
              )}
            </div>
          </div>
          <button className="theme-toggle" onClick={() => setTheme(theme === "light" ? "dark" : "light")} title="Đổi giao diện">
            {theme === "light" ? "☾" : "☀"}
          </button>
        </div>

        <div className="chat-scroll" ref={scrollRef}>
          <div className="chat-inner">
            {!active || active.messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-seal">L</div>
                <h1>Trợ lý pháp lý Việt Nam</h1>
                <p>
                  Đặt câu hỏi về doanh nghiệp, lao động, thuế, sở hữu trí tuệ… Hệ thống sẽ tra cứu
                  trực tiếp trong cơ sở dữ liệu pháp luật, trích dẫn điều luật và tổng hợp câu trả
                  lời có căn cứ.
                </p>
              </div>
            ) : (
              active.messages.map((m) => <ChatMessageView key={m.id} msg={m} />)
            )}
          </div>
        </div>

        <Composer onSend={handleSend} busy={busy} />
      </main>
    </div>
  );
}
