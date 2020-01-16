import os

import pandas as pd


def comple_submit_auc(df):
    candidate_aucs = (
        df[["AuctionID", "score"]]
        .groupby("AuctionID", as_index=False).mean().sort_values("score", ascending=False).iloc[:40, :]
    )
    candidate_aucs["score"] = -999
    target_users = df.groupby("KaiinID", as_index=False).count().query("score < 20")["KaiinID"].tolist()
    buf = []
    for user in target_users:
        candidate_aucs_tmp = candidate_aucs.copy()
        candidate_aucs_tmp["KaiinID"] = user
        buf.append(candidate_aucs_tmp)
    df_comple = pd.concat(buf)
    df_colmled = pd.concat([df, df_comple], sort=False)
    return df_colmled


def adjust_sub_form(users, pred, drop=False):
    sub_data = users.merge(pred, on="KaiinID", how="left")[["KaiinID", "AuctionID", "score"]]
    sub_data = comple_submit_auc(sub_data)
    sub_data.sort_values(['KaiinID', 'score'], ascending=[True, False], inplace=True)
    sub_data['rank'] = sub_data.groupby('KaiinID')['score'].cumcount()
    sub_valid = sub_data.query("rank < =19")
    sub_valid = sub_valid.sort_values(['KaiinID', 'score'], ascending=[True, False]).astype(int)
    if drop:
        sub_valid.drop(["score", "rank"], axis=1, inplace=True)

    return sub_valid
