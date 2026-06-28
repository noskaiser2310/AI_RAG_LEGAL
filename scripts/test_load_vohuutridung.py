"""Quick test: load vohuutridung content config with streaming."""
import os
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

ds = load_dataset("vohuutridung/vietnamese-legal-documents", "content", split="data", streaming=True)
for i, row in enumerate(ds):
    print(f"ID: {row['id']}, content_len: {len(row['content'] or '')}")
    if i >= 2:
        break
print("Success!")
