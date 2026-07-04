"""
Run all available benchmarks.

Usage:
    python scripts/run_all_benchmarks.py
    python scripts/run_all_benchmarks.py --max-queries 50
    python scripts/run_all_benchmarks.py --skip-vlqa
"""
import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "pbgdpl-retrieval": "python scripts/evaluate.py --mode retrieval --max-queries {max_q}",
    "pbgdpl-full": "python scripts/evaluate.py --mode full --max-queries {max_q}",
    "vlqa": "python scripts/evaluate_vlqa.py --max-queries {max_q}",
}


async def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--max-queries", type=int, default=200, help="Max queries per benchmark")
    parser.add_argument("--skip-vlqa", action="store_true", help="Skip VLQA if not available")
    args = parser.parse_args()

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, cmd in BENCHMARKS.items():
        if args.skip_vlqa and "vlqa" in name:
            logger.info(f"Skipping {name}")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")

        full_cmd = cmd.format(max_q=args.max_queries)
        try:
            result = subprocess.run(full_cmd, shell=True, capture_output=False, text=True)
            if result.returncode != 0:
                logger.error(f"{name} failed (exit={result.returncode})")
        except Exception as e:
            logger.error(f"{name} error: {e}")

    logger.info("All benchmarks completed!")
    print("\nResults saved to data/results/")
    for f in sorted(results_dir.glob("*metrics*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    asyncio.run(main())
