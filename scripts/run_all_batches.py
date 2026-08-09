"""
Run all VMTEB eval batches sequentially.
Usage: python run_all_batches.py [--start 0] [--end 12]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BATCH_SIZE = 50
WORKERS = 5
TOTAL_QUERIES = 620


def run_batch(batch_idx, llm="gemini", hf_model="Qwen/Qwen2.5-7B-Instruct", hf_4bit=True, workers=None):
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log = results_dir / f"batch{batch_idx}.log"
    n_workers = workers if workers is not None else (1 if llm == "hf" else WORKERS)
    cmd = [
        sys.executable, "-m", "scripts.evaluate_vmteb_batch",
        "--batch-idx", str(batch_idx),
        "--batch-size", str(BATCH_SIZE),
        "--workers", str(n_workers),
        "--llm", llm,
        "--hf-model", hf_model,
        "--hf-4bit" if hf_4bit else "--no-hf-4bit",
    ]
    print(f"\n{'='*60}")
    print(f"BATCH {batch_idx} (queries {batch_idx*BATCH_SIZE}-{min((batch_idx+1)*BATCH_SIZE, TOTAL_QUERIES)-1})")
    print(f"{'='*60}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s, Return code: {result.returncode}")

    # Parse results
    metrics_path = Path("data/results") / f"vmteb_batch{batch_idx}_metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        ok = m.get("num_queries", 0)
        err = m.get("num_errors", 0)
        f2 = m.get("micro_f2", "?")
        r5 = m.get("recall@5", "?")
        print(f"  Result: {ok}/{ok+err} OK, Micro-F2={f2}, Recall@5={r5}")
    else:
        print(f"  Result: FAILED (no metrics file)")
    return result.returncode == 0 or result.returncode is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--llm", choices=["gemini", "hf"], default="gemini")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=None, help="Override workers (default: 1 for hf, 5 for gemini)")
    parser.add_argument("--force", action="store_true", help="Chay lai batch da co ket qua (mac dinh: skip batch da xong)")
    args = parser.parse_args()

    for batch_idx in range(args.start, args.end + 1):
        metrics_path = Path("data/results") / f"vmteb_batch{batch_idx}_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"Batch {batch_idx}: already done, SKIPPING (use --force to rerun)")
            continue
        ok = run_batch(batch_idx, llm=args.llm, hf_model=args.hf_model, hf_4bit=args.hf_4bit, workers=args.workers)
        if not ok:
            print(f"  Batch {batch_idx} failed, continuing...")

    # Summary
    from pathlib import Path
    import json
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = 0
    all_err = 0
    for batch_idx in range(args.start, args.end + 1):
        metrics_path = Path("data/results") / f"vmteb_batch{batch_idx}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            ok = m.get("num_queries", 0)
            err = m.get("num_errors", 0)
            f2 = m.get("micro_f2", "?")
            all_ok += ok
            all_err += err
            print(f"  Batch {batch_idx}: {ok}/{ok+err} OK, F2={f2}")
        else:
            print(f"  Batch {batch_idx}: NO DATA")
    print(f"  Total: {all_ok}/{all_ok+all_err} OK")


if __name__ == "__main__":
    main()

