"""
Package all eval results into a single zip for download after each Kaggle session.
Usage: python scripts/package_results.py [--out data/results] [--name vmteb_results]
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import config  # noqa: E402


def build_summary(results_dir: Path) -> Path | None:
    metrics_files = sorted(results_dir.glob("vmteb_batch*_metrics.json"))
    if not metrics_files:
        return None
    rows = []
    for f in metrics_files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "batch": m.get("batch_idx", f.stem),
            "num_queries": m.get("num_queries", 0),
            "num_errors": m.get("num_errors", 0),
            "macro_f2": m.get("macro_f2"),
            "micro_f2": m.get("micro_f2"),
            "micro_precision": m.get("micro_precision"),
            "micro_recall": m.get("micro_recall"),
            "mrr": m.get("mrr"),
            "recall@5": m.get("recall@5"),
            "avg_time": m.get("avg_time"),
        })
    if not rows:
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        out = results_dir / "vmteb_summary.csv"
        df.to_csv(out, index=False)
        return out
    except ImportError:
        out = results_dir / "vmteb_summary.csv"
        with open(out, "w", encoding="utf-8") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in rows[0]) + "\n")
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(config.DATA_DIR) / "results"))
    parser.add_argument("--name", default="vmteb_results")
    args = parser.parse_args()

    results_dir = Path(args.out)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        sys.exit(1)

    build_summary(results_dir)

    patterns = ["vmteb_batch*", "vmteb_summary.csv", "batch*.log", "*.jsonl"]
    files = []
    seen = set()
    for pat in patterns:
        for f in sorted(results_dir.glob(pat)):
            if f.is_file() and str(f) not in seen:
                seen.add(str(f))
                files.append(f)

    if not files:
        print(f"WARN: no result files in {results_dir}")
        sys.exit(0)

    ts = time.strftime("%Y%m%d_%H%M%S")
    zip_path = results_dir / f"{args.name}_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    total = sum(f.stat().st_size for f in files)
    print(f"\n{'='*50}")
    print(f"Packaged {len(files)} files ({total/2**20:.1f} MB -> {zip_path.stat().st_size/2**20:.1f} MB zip)")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size/2**20:.2f} MB)")
    print(f"\nZIP: {zip_path}")
    print("Download: open the file explorer in Kaggle (left panel -> Output/Data) and download.")


if __name__ == "__main__":
    main()