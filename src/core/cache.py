"""HF cache resolution: ưu tiên snapshot đã có local, fallback repo name.

Tránh transformers gọi remote mỗi lần khởi động khi model đã nằm trong
HF_HOME (ví dụ .no_exist markers gây lỗi resolve qua tên repo).
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def hf_cache_snapshot(repo_id: str) -> str | None:
    """Trả về snapshot path nếu repo đã nằm trong HF_HOME cache."""
    hf_home = Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None
    candidates = []
    if hf_home:
        candidates.append(hf_home / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    candidates.append(Path(os.environ.get("HF_HUB_CACHE", "")))
    candidates.append(Path(os.environ.get("HF_HOME", "")) / "hub")

    name = repo_id.replace("/", "--")
    for base in candidates:
        if not base:
            continue
        root = base / f"models--{name}"
        if not root.exists():
            continue
        snapshots = root / "snapshots"
        if not snapshots.is_dir():
            continue
        entries = [p for p in snapshots.iterdir() if p.is_dir()]
        if entries:
            snapshot = max(entries, key=lambda p: p.stat().st_mtime)
            logger.info(f"HF cache hit: {repo_id} -> {snapshot}")
            return str(snapshot)
    return None


def resolve_model_path(repo_id: str) -> str:
    """Trả về snapshot path nếu có cache, ngược lại trả về repo_id
    (transformers sẽ download khi cần)."""
    snapshot = hf_cache_snapshot(repo_id)
    return snapshot or repo_id
