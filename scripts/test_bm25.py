"""Test riêng BM25 sparse retrieval (không phụ thuộc embedding/dense)."""
import os, sys, json, time
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, "D:\\AI_RAG_LEGAL")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from src.retrieval.indexing import SparseIndex
from src.retrieval.text_processor import normalize_vietnamese_light, expand_legal_abbreviations

CORPUS = "D:\\AI_RAG_LEGAL\\data\\processed\\corpus.jsonl"
SPARSE = "D:\\AI_RAG_LEGAL\\data\\indexes\\sparse"
QUERIES = [
    "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
    "Hộ kinh doanh cần đáp ứng điều kiện gì để được cấp giấy chứng nhận đăng ký?",
    "Người lao động có được hưởng lương trong thời gian thử việc không?",
    "Mức xử phạt khi công ty giữ bản chính bằng cấp của nhân viên?",
]

def build_offsets(path):
    offs = []
    with open(path, "rb") as f:
        off = f.tell(); ln = f.readline()
        while ln:
            offs.append(off); off = f.tell(); ln = f.readline()
    return offs

def main():
    t0 = time.time(); offsets = build_offsets(CORPUS)
    print(f"Corpus offsets: {len(offsets):,} dòng ({time.time()-t0:.1f}s)")
    def get_doc(i):
        with open(CORPUS, "rb") as f:
            f.seek(offsets[i]); return json.loads(f.readline())

    t0 = time.time(); si = SparseIndex(); si.load(SPARSE)
    print(f"BM25 loaded: {len(si.doc_ids):,} docs ({time.time()-t0:.1f}s)\n")

    for qi, q in enumerate(QUERIES):
        prep = expand_legal_abbreviations(normalize_vietnamese_light(q))
        t0 = time.time(); hits = si.search(prep, 10); dt = time.time() - t0
        print("=" * 100)
        print(f"[Q{qi+1}] {q}   [{dt*1000:.0f}ms]")
        print("=" * 100)
        for r, (idx, score) in enumerate(hits[:8]):
            d = get_doc(int(idx))
            sk = d.get("so_ky_hieu") or d.get("title")
            print(f"  {r+1}. [bm25={score:.2f}] {sk} | Điều {d.get('article_id')} ({d.get('source')})")
            print(f"     {d.get('text','')[:140].replace(chr(10),' ')}")
        print()

if __name__ == "__main__":
    main()
