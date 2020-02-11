# 入荷お知らせに設定していたブランド、オークションか
# 入荷お知らせが何回来たか

nyuuka_oshirase_arranged = nyuuka_oshirase[nyuuka_oshirase["CreateDate"] <= oldest_dtime]

for col_set in [["KaiinID"], ["KaiinID", "BrandID"], ["KaiinID", "CategoryID"]]:
    col_prefix = "_".join(col_set)
    nyuuka_oshirase_cross = cross_counts(df=nyuuka_oshirase_arranged.dropna(subset=col_set),
                                         col_set=col_set, col_name=f"{col_prefix}_nyuuka_cnt")
    dataset_base_b = dataset_base_b.merge(nyuuka_oshirase_cross, on=col_set, how="left").fillna(0)

# 入荷お知らせに設定していたブランド、オークションか
# 入荷お知らせが何回来たか
search_log_arranged = search_log[search_log["TourokuTime"]<=oldest_dtime]

for col_set in [["KaiinID"], ["KaiinID", "BrandID"], ["KaiinID", "CategoryID"]]:
    col_prefix = "_".join(col_set)
    search_log_cross = cross_counts(df=search_log_arranged.dropna(subset=col_set),
                 col_set=col_set, col_name=f"{col_prefix}_search_cnt")
    dataset_base_b = dataset_base_b.merge(search_log_cross, on=col_set, how="left").fillna(0)
