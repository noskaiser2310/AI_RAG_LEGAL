import type { ChatResponse, HealthInfo } from "./types";

const BASE = "/api";

export async function getHealth(): Promise<HealthInfo> {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function postChat(
  query: string,
  mode: string,
  useSelfCorrect: boolean,
): Promise<ChatResponse> {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode,
      use_self_correct: useSelfCorrect,
      max_iterations: 2,
    }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

export async function streamChat(
  query: string,
  mode: string,
  useSelfCorrect: boolean,
  onDone: (result: ChatResponse) => void,
  onError: (err: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode,
      use_self_correct: useSelfCorrect,
      max_iterations: 2,
    }),
    signal,
  });
  if (!r.ok || !r.body) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${r.status}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const lines = block.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        if (line.startsWith("data: ")) data += line.slice(6).trim();
      }
      if (!data) continue;
      if (event === "done") {
        onDone(JSON.parse(data) as ChatResponse);
      } else if (event === "error") {
        const parsed = JSON.parse(data) as { detail: string };
        onError(parsed.detail || "Lỗi không xác định");
      }
    }
  }
}
