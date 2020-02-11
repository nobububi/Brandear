import re

import pandas as pd


def add_datepart(df: pd.DataFrame, field_name: str,
                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55
    dtのカラム(field_name)から年月、月初などの特徴量を作成する関数
    """
    df_datepart = df.copy()
    field = df_datepart[field_name]
    prefix = re.sub('[Dd]ate$', '', field_name)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if time:
        attr = attr + ['Hour', 'Minute']
    for n in attr:
        df_datepart[prefix + n] = getattr(field.dt, n.lower())
    if drop:
        df_datepart.drop(field_name, axis=1, inplace=True)

    return df_datepart


def add_features(dataset_base, watch, bid, auction, period):
    # 特徴量追加部分
    dataset = dataset_base.merge(auction, on="AuctionID")
    watch_auc = watch.merge(auction, on="AuctionID")
    bid_auc = bid.merge(auction, on="AuctionID")

    col_sets = [["KaiinID", "ShouhinID"], ["KaiinID", "BrandID"],
                ["KaiinID", "GenreGroupID"], ["KaiinID", "LineID"]]

    dataset = add_value_counts(
        dataset=dataset,
        feature_df=watch_auc,
        colsets=col_sets,
        prefix="watch",
        oldest_dtime=period["oldest"],
        time_col="TourokuDate"
    )
    
    dataset = add_value_counts(
        dataset=dataset,
        feature_df=bid_auc,
        colsets=col_sets,
        prefix="bid",
        oldest_dtime=period["oldest"],
        time_col="ShudouNyuusatsuDate"
    )

    drop_cols = ["ShouhinShubetsuID", "ShouhinID", "BrandID", "GenreID", "GenreGroupID", "LineID"]
    dataset = dataset.drop(drop_cols, axis=1).fillna(-1)

    return dataset


def add_features(dataset_base, watch, bid, auction, period):
    # 特徴量追加部分
    dataset = dataset_base.merge(auction, on="AuctionID")
    watch_auc = watch.merge(auction, on="AuctionID")
    bid_auc = bid.merge(auction, on="AuctionID")

    # dataset = add_time_features(dataset, watch_auc, "TourokuDate", "watch", period["oldest"])
    # dataset = add_time_features(dataset, bid_auc, "ShudouNyuusatsuDate", "bid", period["oldest"])

    col_sets = [["AuctionID"], ["ShouhinID"], ["BrandID"], ["LineID"], ["KaiinID", "ShouhinID"],
                         ["KaiinID", "BrandID"], ["KaiinID", "GenreGroupID"], ["KaiinID", "LineID"]]
    dataset = add_value_counts(
        dataset=dataset,
        feature_df=watch_auc,
        colsets=col_sets,
        prefix="watch",
        oldest_dtime=period["oldest"],
        time_col="TourokuDate"
    )
    dataset = add_value_counts(
        dataset=dataset,
        feature_df=bid_auc,
        colsets=col_sets,
        prefix="bid",
        oldest_dtime=period["oldest"],
        time_col="ShudouNyuusatsuDate"
    )

    drop_cols = ["ShouhinShubetsuID", "ShouhinID", "BrandID", "GenreID", "GenreGroupID", "LineID"]
    dataset = dataset.drop(drop_cols, axis=1).fillna(-1)

    return dataset

def add_time_features(dataset, feature_df, time_col, prefix, oldest_dtime):
    tmp_time_col = f"Tmp{time_col}Delta"
    key_cols = ["KaiinID", "AuctionID"]
    feature_df[tmp_time_col] = feature_df[time_col].apply(lambda d: (oldest_dtime - d).days)
    time_features = (
        feature_df
        .groupby(key_cols)[tmp_time_col]
        .agg(["count", "max", "min"])
        .rename(columns={"count": f"{prefix}_ua_cnt", "max": f"{prefix}_ua_newest", "min": f"{prefix}_ua_oldest"})
    )
    time_features[f"{prefix}_period"] = time_features[f"{prefix}_ua_newest"] - time_features[f"{prefix}_ua_oldest"]
    output = dataset.merge(time_features, on=key_cols, how="left")
    return output


# def add_value_counts(dataset, feature_df, colsets, prefix, oldest_dtime, time_col):
#     df_cp = dataset.copy()
#     feature_valid = feature_df[feature_df[time_col] < oldest_dtime]
#     for colset in colsets:
#         if len(colset) == 2:
#             cnts = (
#                 feature_valid[colset + ["AuctionID"]].groupby(colset, as_index=False).count()
#                 .rename(columns={"AuctionID": f"{prefix}_{colset[0]}_{colset[1]}_cnt"})
#             )
#         elif len(colset) == 1:
#             col = colset[0]
#             cnts = feature_valid[col].value_counts().reset_index().rename(
#                 columns={"index": col, col: f"{prefix}_{col}_cnt"})
#
#         df_cp = df_cp.merge(cnts, on=colset, how="left")
#     return df_cp


def cross_counts(df, col_set):
    if isinstance(col_set, str):
        cnt_col_name = col_set + "_cnt"
    elif isinstance(col_set, list):
        cnt_col_name = "_".join(col_set) + "_cnt"
    cnts = (
        df.groupby(col_set, as_index=False).size().reset_index()
        .rename(columns={0: cnt_col_name})
    )
    return cnts


def merge_features(df1, df2, key, prefix):
    df2.columns = [prefix + "_" + column if column not in key else column
                   for column in df2.columns]
    return df1.merge(df2, on=key, how="left")


def add_cross_counts(df, feature_df, prefix, col_sets):
    print("##################")
    print("start cross count")
    df_copy = df.copy()
    print(col_sets)
    for col_set in col_sets:
        print(col_set)
        cnts = cross_counts(df=feature_df, col_set=col_set)
        df_copy = merge_features(df_copy, cnts, key=col_set, prefix=prefix).fillna(0)
    return df_copy
