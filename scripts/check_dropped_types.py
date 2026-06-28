"""Check if dropped types could contain useful documents."""
import os, sys
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq
from pathlib import Path

cache_dir = Path("D:\\AI_RAG_LEGAL\\hf_cache\\hub\\datasets--vohuutridung--vietnamese-legal-documents\\snapshots")
pq_file = list((list(cache_dir.iterdir())[0] / "metadata").glob("*.parquet"))[0]
df = pq.read_table(pq_file).to_pandas()

# Check "risky" dropped types - do any have normative doc numbers?
risky_types = ["Điều ước quốc tế", "Sắc lệnh", "Lệnh", "Sắc luật", "Văn bản khác", "Điều ước", "Điều lệ"]

normative_patterns = ["/NĐ-", "/TT-", "/QH", "/TTLT", "/PL-", "/NQ-", "/CT-"]

for t in risky_types:
    sub = df[df["legal_type"] == t]
    has_norm = sub["document_number"].apply(
        lambda x: any(p in str(x) for p in normative_patterns) if isinstance(x, str) else False
    ).sum()
    print(f"{t}: {len(sub)} total, {has_norm} with normative pattern")
    if has_norm > 0:
        mask = sub["document_number"].apply(
            lambda x: any(p in str(x) for p in normative_patterns) if isinstance(x, str) else False
        )
        print(f"  Examples: {sub[mask]['document_number'].head(5).tolist()}")

# Check: do Công văn / Thông báo / Kế hoạch ever have "Điều" in content?
# We can't check content from metadata, but we can check titles
for t in ["Công văn", "Thông báo", "Kế hoạch", "Báo cáo", "Công điện"]:
    sub = df[df["legal_type"] == t]
    has_dieu = sum(1 for title in sub["title"].dropna() if "Điều" in str(title))
    print(f"\n{t}: {has_dieu}/{len(sub)} có 'Điều' trong title")

# For Quyết định, what are those without /QĐ-?
qđ = df[df["legal_type"] == "Quyết định"]
qđ_no_qd = qđ[~qđ["document_number"].str.contains("/QĐ-", na=False)]
print(f"\nQuyết định không có /QĐ-: {len(qđ_no_qd)}")
print(f"  Trong đó có /NĐ-: {(qđ_no_qd['document_number'].str.contains('/NĐ-', na=False)).sum()}")
print(f"  Trong đó có /TT-: {(qđ_no_qd['document_number'].str.contains('/TT-', na=False)).sum()}")
print(f"  Có 'Điều' trong title: {sum(1 for t in qđ_no_qd['title'].dropna() if 'Điều' in str(t))}")
print(f"  Issuing authorities:")
for a, c in qđ_no_qd["issuing_authority"].value_counts().head(10).items():
    print(f"    {a}: {c}")

# Điều ước quốc tế - chi tiết
duqt = df[df["legal_type"] == "Điều ước quốc tế"]
print(f"\nĐiều ước quốc tế issuing authorities:")
for a, c in duqt["issuing_authority"].value_counts().head(10).items():
    print(f"  {a}: {c}")
print(f"Mẫu title:", duqt["title"].head(10).tolist())

# Văn bản khác - chi tiết  
vbk = df[df["legal_type"] == "Văn bản khác"]
print(f"\nVăn bản khác - issuing authorities:")
for a, c in vbk["issuing_authority"].value_counts().head(10).items():
    print(f"  {a}: {c}")
print(f"Mẫu title:", vbk["title"].head(10).tolist())
