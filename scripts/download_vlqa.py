"""
Download VLQA benchmark dataset from available sources.

VLQA: 3,129 questions, 59,636 articles (arXiv:2507.19995)

Usage:
    python scripts/download_vlqa.py                          # auto-detect
    python scripts/download_vlqa.py --output data/raw/vlqa.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

HF_CANDIDATES = [
    ("vlqa/vlqa", "test"),
    ("JAIST/vlqa", "test"),
    ("minhnt/vlqa", "test"),
    ("vitnamese-legal/vlqa", "test"),
    ("UET/vlqa", "test"),
]


def main():
    parser = argparse.ArgumentParser(description="Download VLQA benchmark")
    parser.add_argument("--output", default="data/raw/vlqa.json", help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = None
        for name, split in HF_CANDIDATES:
            print(f"Trying: {name} ({split})...")
            try:
                ds = load_dataset(name, split=split, streaming=True)
                print(f"Found: {name}")
                break
            except Exception:
                continue

        if ds is not None:
            data = []
            for i, row in enumerate(ds):
                item = {
                    "id": i,
                    "question": row.get("question", row.get("question_text", "")),
                    "relevant_articles": row.get("relevant_articles", row.get("reference_articles", [])),
                    "answer": row.get("answer", row.get("answer_text", "")),
                }
                data.append(item)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} queries to {output_path}")
            return
    except ImportError:
        print("datasets not installed. Try: pip install datasets")

    print()
    print("VLQA not found on HuggingFace.")
    print("Please download manually from:")
    print("  - Paper: https://arxiv.org/abs/2507.19995")
    print("  - HF:    https://huggingface.co/datasets?search=vlqa")
    print()
    print("Then convert to JSON format:")
    print('  [{"id": 0, "question": "...", "relevant_articles": ["..."]}]')
    sys.exit(1)


if __name__ == "__main__":
    main()
