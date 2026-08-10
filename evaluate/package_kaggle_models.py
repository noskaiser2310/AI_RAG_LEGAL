"""
Đóng gói HF cache (model + dataset eval) thành Kaggle Dataset để benchmark TRÊN Kaggle.

Chỉ đóng gói các model/dataset đã có trong hf_cache local:
  - models--mainguyen9--vietlegal-harrier-0.6b      (embedding, ~2.2GB)
  - models--AITeamVN--Vietnamese_Reranker           (reranker, ~2.1GB)
  - datasets--another-symato--VMTEB-ALQAC-retrieval (eval, ~0.1MB)

KHÔNG đóng gói LLM local (Qwen) — tải nhanh trên Kaggle, giảm dung lượng local.

Usage:
  python scripts/package_kaggle_models.py            # tạo kaggle_assets/models_pkg/
  python scripts/package_kaggle_models.py --push     # + upload len Kaggle

Cau truc output: kaggle_assets/models_pkg/hub/{models,d datasets}--*/...
Notebook maple/img/cell: copy => /kaggle/working/hf_cache/hub/* de HF doc model tu cache.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_HUB = ROOT / "hf_cache" / "hub"
PKG_DIR = ROOT / "kaggle_assets" / "models_pkg"

# model/dataset cache can gói (tên theo layout HF hub: models--owner--name, datasets--owner--name)
INCLUDES = [
    "models--mainguyen9--vietlegal-harrier-0.6b",
    "models--AITeamVN--Vietnamese_Reranker",
    "datasets--another-symato--VMTEB-ALQAC-retrieval",
]


def collect():
    result = []
    for name in INCLUDES:
        src = HF_HUB / name
        if not src.exists():
            print(f"  SKIP {name} (chua co trong hf_cache)")
            continue
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        result.append((name, size))
        # luôn bỏ .locks khi gói (mỗi session trống)
        for f in src.rglob(".locks"):
            if f.is_dir():
                print(f"  removing {name}/.locks")
                # don't actually remove source; fine — locks không được copy
        print(f"  INCLUDE {name}: {size/1e9:.2f} GB")
    return result


def make_pkg():
    if not HF_HUB.exists():
        raise SystemExit(f"HF_HUB khong ton tai: {HF_HUB}")
    shutil.rmtree(PKG_DIR, ignore_errors=True)
    for name in INCLUDES:
        src = HF_HUB / name
        if not src.exists():
            continue
        dst = PKG_DIR / "hub" / name
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".locks", "*.lock"))
        print(f"  copied hub/{name}")
    # tổng size
    total = sum(f.stat().st_size for f in PKG_DIR.rglob("*") if f.is_file())
    print(f"\nPkg: {PKG_DIR} ({total/1e9:.2f} GB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    collect()
    make_pkg()

    if args.push:
        # upload dataset qua Kaggle CLI (cần dataset-metadata.json trong PKG_DIR)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create",
                        "-p", str(PKG_DIR)], check=False)

    print("\nNext:")
    print("  1. dataset 'ai-rag-legal-models' can chieu theo Kaggle")
    print("  2. notebook: Add Input -> ai-rag-legal-models")
    print("  3. Cell model: copy /kaggle/input/.../hub/* vao /kaggle/working/hf_cache/hub/ (code se them)")