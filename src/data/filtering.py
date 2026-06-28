"""
Smart filtering of legal documents for Vietnamese Legal RAG.

Research-based filtering strategy (verified against 518K vohuutridung metadata):

TIER_KEEP = definitely normative or may contain normative content
  - Luật, Bộ luật, Hiến pháp, Pháp lệnh: core legislation (~827)
  - Nghị định (~5.7K), Thông tư (~16.7K), TTLT (~2.6K): detailed guidance
  - Văn bản hợp nhất (~3.6K): consolidated legal texts
  - Nghị quyết (~41K): 96% normative format, some have articles
  - Quyết định (~249K): 99.9% individual but ~4.8K without /QD- are potentially normative
  - Chỉ thị (~14K), Hướng dẫn (~2K): mixed, some contain articles
  - Quy định (~325), Quy chế (~194): normative rules
  - Văn bản khác (~1.1K): contains "Kết luận của Ban Chấp hành Trung ương" (important!)
  - Điều ước quốc tế (~1.3K): international treaties, potentially relevant
  - Điều ước (~6), Điều lệ (~16): misc normative

TIER_DROP = definitively non-normative, zero articles in content
  - Công văn (~124K): administrative correspondence (0 articles)
  - Kế hoạch (~32K): operational plans (0 articles)
  - Thông báo (~15K): notifications (0 articles)
  - Báo cáo (~1.5K), Công điện (~1.7K): reports, telegrams
  - Tiêu chuẩn VN/ngành/XDVN (~19): technical standards
  - WTO_* (~61): trade-specific WTO documents
  - Thông tri (~37): historical circular
  - Sắc lệnh (~990), Lệnh (~641), Sắc luật (~5): pre-1975 historical
"""

TIER_KEEP = {
    "Luật", "Bộ luật", "Hiến pháp", "Pháp lệnh",
    "Nghị định", "Thông tư", "Thông tư liên tịch",
    "Văn bản hợp nhất", "Nghị quyết",
    "Chỉ thị", "Hướng dẫn", "Quyết định",
    "Quy định", "Quy chế",
    "Văn bản khác",
    "Điều ước quốc tế", "Điều ước", "Điều lệ",
}

TIER_DROP = {
    "Công văn", "Kế hoạch", "Thông báo", "Công điện", "Báo cáo",
    "WTO_Văn bản", "WTO_Cam kết VN",
    "Công ước", "Hiệp định", "Thoả thuận", "Nghị định thư",
    "Sắc lệnh", "Sắc luật", "Lệnh",
    "Tiêu chuẩn Việt Nam", "Tiêu chuẩn ngành", "Tiêu chuẩn XDVN",
    "Thông tri", "Văn bản WTO",
}


def should_keep_by_type(legal_type: str) -> bool:
    """Fast metadata-level filter: should we even load content for this doc type?"""
    return legal_type in TIER_KEEP


def filter_vohuutridung_metadata(meta_df: "pd.DataFrame") -> "pd.DataFrame":
    """Filter vohuutridung metadata by legal_type (fast metadata-only pass)."""
    import pandas as pd  # noqa: F811
    before = len(meta_df)
    meta_df = meta_df[meta_df["legal_type"].isin(TIER_KEEP)].copy()
    after = len(meta_df)
    dropped = before - after
    print(f"[Filter] vohuutridung: {before} -> {after} docs (dropped {dropped} TIER_DROP)")
    return meta_df
