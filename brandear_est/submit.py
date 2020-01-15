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
    df_colmled = pd.concat([df, df_comple])
    return df_colmled