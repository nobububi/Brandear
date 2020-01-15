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


def add_features(dataset, watch, bid, oldest_dtime):
    # 特徴量追加部分

    dataset = add_time_features(dataset, watch, "TourokuDate", "watch", oldest_dtime)
    dataset = add_value_counts(
        dataset, watch, [["AuctionID"], ["ShouhinID"], ["BrandID"], ["LineID"], ["KaiinID", "ShouhinID"],
                         ["KaiinID", "BrandID"], ["KaiinID", "GenreGroupID"], ["KaiinID", "LineID"]], "watch"
    )

    dataset = add_time_features(dataset, bid, "ShudouNyuusatsuDate", "bid", oldest_dtime)
    dataset = add_value_counts(
        dataset, bid, [["AuctionID"], ["ShouhinID"], ["BrandID"], ["LineID"], ["KaiinID", "ShouhinID"],
                       ["KaiinID", "BrandID"], ["KaiinID", "GenreGroupID"], ["KaiinID", "LineID"]], "bid"
    )

    drop_cols = ["ShouhinShubetsuID", "ShouhinID", "BrandID", "GenreID", "GenreGroupID", "LineID"]

    dataset = dataset.drop(drop_cols, axis=1).fillna(-1)

    return dataset


def add_time_features(df, feature_df, time_col, prefix, oldest_dtime):
    tmp_time_col = f"Tmp{time_col}Delta"
    key_cols = ["KaiinID", "AuctionID"]
    feature_df.loc[tmp_time_col] = feature_df[time_col].apply(lambda d: (oldest_dtime - d).days)
    time_features = (
        feature_df
            .groupby(key_cols)[tmp_time_col]
            .agg(["count", "max", "min"])
            .rename(columns={"count": f"{prefix}_ua_cnt", "max": f"{prefix}_ua_newest", "min": f"{prefix}_ua_oldest"})
    )
    time_features[f"{prefix}_period"] = time_features[f"{prefix}_ua_newest"] - time_features[f"{prefix}_ua_oldest"]
    output = df.merge(time_features, on=key_cols, how="left")
    return output


def add_value_counts(df, feature_df, colsets, prefix):
    df_cp = df.copy()
    for colset in colsets:
        if len(colset) == 2:
            cnts = (
                feature_df[colset + ["AuctionID"]].groupby(colset, as_index=False).count()
                    .rename(columns={"AuctionID": f"{prefix}_{colset[0]}_{colset[1]}_cnt"})
            )
        elif len(colset) == 1:
            col = colset[0]
            cnts = feature_df[col].value_counts().reset_index().rename(
                columns={"index": col, col: f"{prefix}_{col}_cnt"})

        df_cp = df_cp.merge(cnts, on=colset, how="left")
    return df_cp

