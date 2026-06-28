"""Sanity-check pooling: embedding phải PHÂN BIỆT được text khác nhau (không collapse)."""
import os, sys, asyncio
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, "D:\\AI_RAG_LEGAL")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from src.embedding.harrier_embedding import HarrierEmbedding

E5 = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó"
query = "Người lao động có được hưởng lương trong thời gian thử việc không?"
relevant = "Tiền lương của người lao động trong thời gian thử việc do hai bên thỏa thuận nhưng ít nhất phải bằng 85% mức lương của công việc đó."
irrelevant = "Vùng nước cảng biển Quảng Ngãi thuộc địa phận tỉnh Quảng Ngãi bao gồm khu vực Dung Quất."

emb = HarrierEmbedding()
qv, rv, iv = [np.array(x) for x in asyncio.run(emb.embed([f"{E5}\nTruy vấn: {query}", relevant, irrelevant]))]

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"cos(query, RELEVANT)   = {cos(qv, rv):.4f}   <- phải CAO")
print(f"cos(query, IRRELEVANT) = {cos(qv, iv):.4f}   <- phải THẤP hơn")
print(f"cos(RELEVANT, IRREL)   = {cos(rv, iv):.4f}")
print()
print("=> Pooling OK" if cos(qv, rv) > cos(qv, iv) + 0.03 else "=> VẪN NGHI NGỜ (chênh lệch quá nhỏ)")

# Hai query khác nhau phải cho vector khác nhau (test chống collapse cũ)
q2 = "Hộ kinh doanh cần điều kiện gì để đăng ký?"
a, b = [np.array(x) for x in asyncio.run(emb.embed([f"{E5}\nTruy vấn: {query}", f"{E5}\nTruy vấn: {q2}"]))]
print(f"\ncos(2 query khác nhau)  = {cos(a, b):.4f}   <- phải < 0.99 (cũ bị ~1.0 do collapse)")
