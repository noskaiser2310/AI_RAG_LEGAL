import React from "react";

function inline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const regex = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const token = m[0];
    if (token.startsWith("**")) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      nodes.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    last = m.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function renderBlocks(md: string): React.ReactNode[] {
  const lines = md.split("\n");
  const out: React.ReactNode[] = [];
  let key = 0;
  let i = 0;

  const flushList = (start: number, ordered: boolean) => {
    const items: string[] = [];
    let j = start;
    while (j < lines.length) {
      const line = lines[j];
      const m = ordered
        ? /^\s*\d+[.)]\s+(.*)$/.exec(line)
        : /^\s*[-*]\s+(.*)$/.exec(line);
      if (!m) break;
      items.push(m[1]);
      j++;
    }
    const Tag = ordered ? "ol" : "ul";
    out.push(
      <Tag key={key++}>
        {items.map((it, idx) => (
          <li key={idx}>{inline(it)}</li>
        ))}
      </Tag>,
    );
    return j;
  };

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const Tag = (`h${level}`) as keyof React.JSX.IntrinsicElements;
      out.push(<Tag key={key++}>{inline(heading[2])}</Tag>);
      i++;
      continue;
    }

    const hr = /^\s*(---+|\*\*\*+)\s*$/.exec(line);
    if (hr) {
      out.push(<hr key={key++} />);
      i++;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      i = flushList(i, false);
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      i = flushList(i, true);
      continue;
    }

    const blockquote = /^\s*>\s?(.*)$/.exec(line);
    if (blockquote) {
      const quoted: string[] = [];
      while (i < lines.length) {
        const q = /^\s*>\s?(.*)$/.exec(lines[i]);
        if (!q) break;
        quoted.push(q[1]);
        i++;
      }
      out.push(<blockquote key={key++}>{inline(quoted.join(" "))}</blockquote>);
      continue;
    }

    const para: string[] = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !/^\s*[-*]\s+/.test(lines[i]) && !/^\s*\d+[.)]\s+/.test(lines[i]) && !/^#{1,4}\s/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    out.push(<p key={key++}>{inline(para.join(" "))}</p>);
  }
  return out;
}

export default function Markdown({ text }: { text: string }) {
  return <div className="answer">{renderBlocks(text)}</div>;
}
