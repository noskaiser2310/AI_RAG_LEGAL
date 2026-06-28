"""Analyze legal type importance for legal QA."""
import os, sys
os.environ["HF_HOME"] = "D:\\AI_RAG_LEGAL\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from pathlib import Path

cache_dir = Path("D:\\AI_RAG_LEGAL\\hf_cache\\hub\\datasets--vohuutridung--vietnamese-legal-documents\\snapshots")
snapshots = list(cache_dir.iterdir())
meta_dir = snapshots[0] / "metadata"
pq_files = list(meta_dir.glob("*.parquet"))
table = pq.read_table(pq_files[0])
df = table.to_pandas()

print(f"Total docs: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Analyze doc_number prefix (normative docs have format like 145/2020/NĐ-CP)
def classify_doc_number(num):
    if not num or not isinstance(num, str):
        return "MISSING"
    if "/NĐ-CP" in num or "/NĐ-" in num:
        return "Nghị định"
    if "/TT-" in num or "/TTLT" in num or "/TT-B" in num:
        return "Thông tư"
    if "/QH" in num or "/QH14" in num or "/QH15" in num:
        return "Luật"
    if "/CT-" in num:
        return "Chỉ thị"
    if "/QĐ-" in num:
        return "Quyết định"
    if "/PL-" in num or "/PL-UBTVQH" in num:
        return "Pháp lệnh"
    if "/NQ-" in num:
        return "Nghị quyết"
    if "/HD-" in num:
        return "Hướng dẫn"
    return "OTHER"

df["num_type"] = df["document_number"].apply(classify_doc_number)

# Distribution by doc_number pattern vs legal_type
cross = pd.crosstab(df["num_type"], df["legal_type"].fillna("UNKNOWN"))
print("\nCross-tab: doc_number pattern vs legal_type")
print(cross.to_string())

# Check: what legal_types have articles (Điều)?
# For this we need content - but let's check doc_type patterns
# that typically have articles
normative_patterns = ["/NĐ-", "/TT-", "/QH", "/TTLT", "/PL-", "/NQ-"]
df["is_normative_num"] = df["document_number"].apply(
    lambda x: any(p in str(x) for p in normative_patterns) if isinstance(x, str) else False
)

print(f"\nNormative doc_number pattern: {df['is_normative_num'].sum()}/{len(df)}")

# Legal types that have normative document numbers
norm_by_type = df[df["is_normative_num"]].groupby("legal_type").size().sort_values(ascending=False)
print("\nLegal types with normative doc numbers:")
for t, c in norm_by_type.items():
    total = len(df[df["legal_type"] == t])
    print(f"  {t}: {c}/{total} ({c/max(total,1)*100:.1f}%)")

# Summary of what to keep
print("\n\n" + "="*60)
print("RECOMMENDED FILTERING STRATEGY")
print("="*60)

tier1 = ["Luật", "Hiến pháp", "Bộ luật", "Pháp lệnh", "Nghị định", 
         "Thông tư", "Thông tư liên tịch", "Văn bản hợp nhất"]
tier2 = ["Nghị quyết", "Chỉ thị", "Hướng dẫn", "Quyết định", "Quy định", "Quy chế"]
tier3 = ["Công văn", "Kế hoạch", "Thông báo", "Công điện", "Báo cáo",
         "Điều ước quốc tế", "WTO_Văn bản", "Công ước", "Hiệp định",
         "Thoả thuận", "Nghị định thư", "Sắc lệnh", "Sắc luật", "Lệnh",
         "Tiêu chuẩn Việt Nam", "Tiêu chuẩn ngành", "Tiêu chuẩn XDVN",
         "Văn bản khác", "Văn bản WTO", "WTO_Cam kết VN",
         "Điều ước", "Điều lệ", "Thông tri"]

t1_count = df[df["legal_type"].isin(tier1)].shape[0]
t2_count = df[df["legal_type"].isin(tier2)].shape[0]
t3_count = df[df["legal_type"].isin(tier3)].shape[0]
other_count = df[~df["legal_type"].isin(tier1 + tier2 + tier3)].shape[0]

print(f"\nTIER 1 (Keep unconditionally): {t1_count} docs ({t1_count/len(df)*100:.1f}%)")
print(f"  Types: {tier1}")
print(f"TIER 2 (Keep only if has articles): {t2_count} docs ({t2_count/len(df)*100:.1f}%)")
print(f"  Types: {tier2}")
print(f"TIER 3 (Drop): {t3_count} docs ({t3_count/len(df)*100:.1f}%)")
print(f"UNCLASSIFIED: {other_count} docs ({other_count/len(df)*100:.1f}%)")

# Save filtered stats
t1_types = df[df["legal_type"].isin(tier1)]["legal_type"].value_counts().to_dict()
t2_types = df[df["legal_type"].isin(tier2)]["legal_type"].value_counts().to_dict()
t3_types = df[df["legal_type"].isin(tier3)]["legal_type"].value_counts().to_dict()

report = {
    "total": len(df),
    "tier1": {"count": int(t1_count), "types": {k: int(v) for k, v in t1_types.items()}},
    "tier2": {"count": int(t2_count), "types": {k: int(v) for k, v in t2_types.items()}},
    "tier3": {"count": int(t3_count), "types": {k: int(v) for k, v in t3_types.items()}},
}
Path("data").mkdir(exist_ok=True)
with open("data/filter_strategy.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nSaved to data/filter_strategy.json")
