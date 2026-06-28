"""Deep dive into TIER 2 document types."""
import os, sys
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq
from pathlib import Path

cache_dir = Path("D:\\AI_RAG_LEGAL\\hf_cache\\hub\\datasets--vohuutridung--vietnamese-legal-documents\\snapshots")
pq_file = list((list(cache_dir.iterdir())[0] / "metadata").glob("*.parquet"))[0]
df = pq.read_table(pq_file).to_pandas()

# Quyết định with normative QD number
qđ_norm = df[(df["legal_type"] == "Quyết định") & df["document_number"].str.contains("/QĐ-", na=False)]
print(f"Quyết định có /QĐ-: {len(qđ_norm)}")
print("Mẫu:", qđ_norm["document_number"].head(10).tolist())

# Nghị quyết
nq = df[df["legal_type"] == "Nghị quyết"]
nq_norm = nq[nq["document_number"].str.contains("/NQ-", na=False)]
nq_other = nq[~nq["document_number"].str.contains("/NQ-", na=False)]
print(f"\nNghị quyết có /NQ-: {len(nq_norm)}")
print(f"Nghị quyết khác: {len(nq_other)}")
print("Mẫu NQ không /NQ-:", nq_other["document_number"].head(10).tolist())

# Chỉ thị
ct = df[df["legal_type"] == "Chỉ thị"]
ct_norm = ct[ct["document_number"].str.contains("/CT-", na=False)]
ct_rest = ct[~ct["document_number"].str.contains("/CT-", na=False)]
print(f"\nChỉ thị có /CT-: {len(ct_norm)}")
print("Mẫu CT khác:", ct_rest["document_number"].head(10).tolist())
print("Issuing authorities:", ct_rest["issuing_authority"].value_counts().head(10).to_dict())

# Hướng dẫn
hd = df[df["legal_type"] == "Hướng dẫn"]
print(f"\nHướng dẫn: {len(hd)}")
print("Mẫu:", hd["document_number"].head(10).tolist())
print("Issuing authorities:", hd["issuing_authority"].value_counts().head(10).to_dict())

# Quy định, Quy chế
for t in ["Quy định", "Quy chế"]:
    sub = df[df["legal_type"] == t]
    print(f"\n{t}: {len(sub)}")
    print("Mẫu:", sub["document_number"].head(5).tolist())

# Issuance years
years = df["issuance_date"].dropna().str[:4]
year_counts = years.value_counts().sort_index()
print("\nNăm ban hành top 10:", year_counts.head(10).to_dict())
print("... bottom 5:", year_counts.tail(5).to_dict())

# Chỉ thị, Hướng dẫn - check if they have articles in content
# Just by title
for t in ["Chỉ thị", "Hướng dẫn"]:
    sub = df[df["legal_type"] == t]
    titles = sub["title"].dropna().tolist()
    has_article_in_title = sum(1 for title in titles if "Điều" in title)
    print(f"\n{t}: {has_article_in_title}/{len(titles)} có 'Điều' trong title")
