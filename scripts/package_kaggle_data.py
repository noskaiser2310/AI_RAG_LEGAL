"""
Đóng gói data (indexes + corpus) thành Kaggle Dataset để benchmark trên Kaggle.

Usage:
  python scripts/package_kaggle_data.py            # tạo kaggle_assets/data_pkg/ + data.zip
  python scripts/package_kaggle_data.py --push     # tạo + đăng lên Kaggle (cần auth)

Sau khi push, notebook kaggle_benchmark_vmteb.ipynb Add Input -> ai-rag-legal-data.
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PKG_DIR = ROOT / "kaggle_assets" / "data_pkg"
ZIP_PATH = ROOT / "kaggle_assets" / "data.zip"

NEEDED = [
    ("data/indexes", "indexes"),
    ("data/processed/corpus.jsonl", "processed/corpus.jsonl"),
]


def collect() -> list[Path]:
    for src, _ in NEEDED:
        p = DATA_DIR / src
        if not p.exists():
            raise SystemExit(f"THIEU: {p} chua ton tai. Chay build index truoc.")
    files: list[Path] = []
    for src, _ in NEEDED:
        p = DATA_DIR / src
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and "__pycache__" not in str(f):
                    files.append(f)
        else:
            files.append(p)
    return files


def make_zip(files):
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in files:
            arc = str(f.relative_to(ROOT)).replace("\\", "/")
            zf.write(f, arc)
            total += f.stat().st_size
    print(f"Created {ZIP_PATH.name}: {total / 1e9:.2f} GB (files: {len(files)})")


def make_pkg_dir(files):
    """Dataset theo cau truc thu muc (khong nen)."""
    shutil.rmtree(PKG_DIR, ignore_errors=True)
    for f in files:
        dst = PKG_DIR / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    print(f"Pkg dir: {PKG_DIR} ({sum(f.stat().st_size for f in files)/1e9:.2f} GB)")


def push_dataset():
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version",
                    "-p", str(PKG_DIR), "-m", "update data"], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload len Kaggle Dataset")
    ap.add_argument("--mode", choices=["zip", "dir", "both"], default="both")
    args = ap.parse_args()

    files = collect()
    print(f"Collected {len(files)} files, {sum(f.stat().st_size for f in files)/1e9:.2f} GB")

    if args.mode in ("zip", "both"):
        make_zip(files)
    if args.mode in ("dir", "both"):
        make_pkg_dir(files)

    if args.push:
        push_dataset()

    print("\nNext: ")
    print("  1. Kaggle -> Dataset -> New dataset -> Upload data.zip")
    print("     hoac: kaggle datasets create -p kaggle_assets/data_pkg/ -t dataset")
    print("  2. Mo notebook kaggle_benchmark_vmteb.ipynb -> Add Input -> ai-rag-legal-data")
    print("  3. Chay all cells")