"""
Zip assets for Kaggle upload.
Khong can API key — tat ca model open-source, chay offline.

Usage:
  python scripts/prepare_kaggle_assets.py        # all
  python scripts/prepare_kaggle_assets.py corpus  # just corpus
  python scripts/prepare_kaggle_assets.py indexes # just indexes
  python scripts/prepare_kaggle_assets.py src     # just source code
"""
import os, zipfile, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEXES_DIR = DATA_DIR / "indexes"
CORPUS_PATH = DATA_DIR / "processed" / "corpus.jsonl"
SRC_DIR = ROOT / "src"
ASSETS_DIR = ROOT / "kaggle_assets"

IGNORE_SRC_DIRS = {"__pycache__"}
NOTEBOOKS = [
    ROOT / "kaggle_pipeline.ipynb",
    ROOT / "kaggle_build_indexes.ipynb",
]
REQUIREMENTS_KAGGLE = ASSETS_DIR / "requirements-kaggle.txt"
REQUIREMENTS_CONTENT = """transformers>=4.48.0
accelerate>=1.3.0
bitsandbytes>=0.43.0
sentencepiece>=0.2.0
datasets>=2.19.0
faiss-gpu>=1.9.0
bm25s>=0.3.0
aiohttp>=3.10.0
"""


def zip_dir(zf, src_dir, arc_prefix):
    for f in sorted(src_dir.rglob("*")):
        if any(p.name in IGNORE_SRC_DIRS for p in f.parents) or f.name.endswith(".pyc"):
            continue
        if f.is_file():
            arcname = f"{arc_prefix}/{f.relative_to(src_dir).as_posix()}"
            size_mb = f.stat().st_size / 1e6
            if size_mb > 10:
                print(f"  {arcname} ({size_mb:.1f} MB)")
            zf.write(f, arcname)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Write requirements file
    with open(REQUIREMENTS_KAGGLE, "w") as f:
        f.write(REQUIREMENTS_CONTENT)
    print(f"Created {REQUIREMENTS_KAGGLE.name}")

    if mode in ("corpus", "all"):
        dst = ASSETS_DIR / "corpus.zip"
        print(f"Creating {dst}...")
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
            size_gb = CORPUS_PATH.stat().st_size / 1e9
            zf.write(CORPUS_PATH, CORPUS_PATH.relative_to(ROOT).as_posix())
            print(f"  corpus.jsonl ({size_gb:.2f} GB)")

    if mode in ("indexes", "all"):
        dst = ASSETS_DIR / "indexes.zip"
        print(f"Creating {dst}...")
        files = [f for f in INDEXES_DIR.rglob("*") if f.is_file()]
        total_gb = sum(f.stat().st_size for f in files) / 1e9
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                arcname = f.relative_to(ROOT).as_posix()
                size_mb = f.stat().st_size / 1e6
                if size_mb > 10:
                    print(f"  {arcname} ({size_mb:.1f} MB)")
                zf.write(f, arcname)
        print(f"  => {total_gb:.2f} GB total")

    if mode in ("src", "all"):
        dst = ASSETS_DIR / "src.zip"
        print(f"Creating {dst}...")
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
            zip_dir(zf, SRC_DIR, "src")
            # Add notebooks
            for nb in NOTEBOOKS:
                if nb.exists():
                    zf.write(nb, nb.relative_to(ROOT).as_posix())
                    print(f"  {nb.name}")
            # Also include requirements for Kaggle
            if REQUIREMENTS_KAGGLE.exists():
                zf.write(REQUIREMENTS_KAGGLE, REQUIREMENTS_KAGGLE.relative_to(ROOT).as_posix())
        total = sum(f.stat().st_size for f in SRC_DIR.rglob("*.py") if "__pycache__" not in str(f))
        print(f"  src/ ({total / 1e6:.1f} MB)")

    print(f"\nAssets in {ASSETS_DIR}/:")
    for f in sorted(ASSETS_DIR.iterdir()):
        if f.is_file() and f.suffix == ".zip":
            print(f"  {f.name} ({f.stat().st_size / 1e9:.2f} GB)")
    print(f"  {REQUIREMENTS_KAGGLE.name}")

    print(f"\nUpload steps:")
    print(f"  1. Go to https://www.kaggle.com/datasets")
    print(f"  2. 'New Dataset' -> name 'ai-rag-legal-assets'")
    print(f"  3. Upload: corpus.zip + indexes.zip + src.zip + requirements-kaggle.txt")
    print(f"  4. Set dataset to Public")
    print(f"  5. Kaggle Notebook: Add Dataset -> ai-rag-legal-assets")
    print(f"  6. Paste kaggle_pipeline.ipynb -> GPU Accelerator (T4x2) -> Run All")
    print(f"  (Khong can API key, khong can Secrets)")


if __name__ == "__main__":
    main()
