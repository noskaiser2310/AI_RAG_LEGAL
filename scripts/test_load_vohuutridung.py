"""Quick test: load vohuutridung content config with streaming."""
import os
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

ds = load_dataset("vohuutridung/vietnamese-legal-documents", "content", split="data", streaming=True)
for i, row in enumerate(ds):
    print(f"ID: {row['id']}, content_len: {len(row['content'] or '')}")
    if i >= 2:
        break
print("Success!")
