"""Vá kaggle_build_indexes.ipynb:
1. Khôi phục newline cho mọi code cell (source list bị mất '\\n').
2. Sửa cell encode dense: CLS pooling -> last-token pooling (khớp HarrierEmbedding).
"""
import json

NB = "D:\\AI_RAG_LEGAL\\kaggle_build_indexes.ipynb"
d = json.load(open(NB, encoding="utf-8"))

OLD_POOL = """        # DataParallel với trust_remote_code model: output có thể là list[ModelOutput] hoặc ModelOutput
        if isinstance(outputs, (list, tuple)):
            emb = torch.cat([o.last_hidden_state[:, 0, :] for o in outputs])
        else:
            emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)"""

NEW_POOL = """        # LAST-TOKEN POOLING (Qwen3 decoder) — phải khớp src/embedding/harrier_embedding.py
        # DataParallel có thể trả list[ModelOutput]; gộp về cuda:0 rồi gather token cuối non-pad
        if isinstance(outputs, (list, tuple)):
            last_hidden = torch.cat([o.last_hidden_state.to('cuda:0') for o in outputs], dim=0)
        else:
            last_hidden = outputs.last_hidden_state
        seq_len = inputs['attention_mask'].sum(dim=1) - 1
        emb = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_len]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)"""

patched_pool = False
for c in d["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c["source"]
    text = "\n".join(src) if isinstance(src, list) else src  # khôi phục newline
    if OLD_POOL in text:
        text = text.replace(OLD_POOL, NEW_POOL)
        patched_pool = True
    # lưu lại dạng list dòng CÓ '\n' (chuẩn nbformat)
    lines = text.split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]

assert patched_pool, "KHÔNG tìm thấy block CLS pooling để thay — kiểm tra lại notebook!"
json.dump(d, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Patched OK: newline fixed + last-token pooling applied.")
