from dateutil.relativedelta import relativedelta

from brandear_est.utils import *


def build_dataset_base(watch, bid, auction, bid_success, dset_type, period, target_users=None):
    if dset_type != "submission":

        # 正解データ作成
        target_actions = extract_target_actions(watch, bid, period)
        target_users = target_actions[["KaiinID"]].drop_duplicates()

        # 正解データとなりうる候補を作成
        target_candidate = build_target_candidate(watch, bid, auction, bid_success, period, target_users)

        # 正解データと正解データ候補を結合
        dataset_base = (
            target_candidate.merge(target_actions, on=["KaiinID", "AuctionID"], how="left")
            .fillna(0)
            .groupby(["KaiinID", "AuctionID"], as_index=False)
            .max()
        )

    elif dset_type == "submission":
        # 正解データとなりうる候補を作成
        dataset_base = build_target_candidate(watch, bid, auction, bid_success, period, target_users)

        # 特徴量付与の対象となるデータ
    return dataset_base


def extract_target_actions(watch, bid, period):
    watch_actioned = (
        watch.loc[(watch["TourokuDate"] >= period["oldest"]) & (watch["TourokuDate"] < period["newest"]),
                  ["KaiinID", "AuctionID"]]
    )
    bid_actioned = (
        bid.loc[(bid["ShudouNyuusatsuDate"] >= period["oldest"]) & (bid["ShudouNyuusatsuDate"] < period["newest"]),
                ["KaiinID", "AuctionID"]]
    )
    # 学習用データの際は正解データを作成する
    watch_actioned["watch_actioned"] = 1
    bid_actioned["bid_actioned"] = 1

    target_actions = (
        watch_actioned
        .merge(bid_actioned, on=["KaiinID", "AuctionID"], how="outer")
        .drop_duplicates()
        .fillna(0)
    )

    return target_actions


def build_target_candidate(watch, bid, auction, bid_success, period, target_users):
    similar_actions = extract_similar_aucs(target_users, auction, bid, watch, period, key="ShouhinID")

    valid_aucs = extract_valid_aucs(auction, watch, bid_success, period)
    sample_num = 20
    aucs_sampled = (
        valid_aucs.sample(n=target_users.shape[0] * sample_num, replace=True)
        .reset_index(drop=True)
    )
    actions_sampled = pd.concat([
        aucs_sampled,
        pd.DataFrame({"KaiinID": target_users["KaiinID"].tolist() * 20})], axis=1, sort=False)

    target_candidate = pd.concat([similar_actions, actions_sampled], sort=False).drop_duplicates()

    return target_candidate


def extract_valid_aucs(auction, watch, bid_success, period):
    valid_aucs = left_anti_join(
        pd.concat([
            auction[(auction["CreateDate"] > period["oldest"] - relativedelta(months=2)) &
                    (auction["CreateDate"] < period["newest"])][["AuctionID"]],
            watch[(watch["TourokuDate"] > period["oldest"] - relativedelta(months=2)) &
                  (watch["TourokuDate"] < period["oldest"])][["AuctionID"]].drop_duplicates()
        ]).drop_duplicates(),
        bid_success[bid_success["RakusatsuDate"] < period["oldest"]], "AuctionID", "AuctionID"
    )
    return valid_aucs


def extract_similar_aucs(target_users, auction, bid, watch, period, key="ShouhinID"):
    users_watch = (
        watch.merge(auction, on="AuctionID", how="inner")
        .merge(target_users, on="KaiinID", how="inner")
    )
    users_bid = (
        bid.merge(auction, on="AuctionID", how="inner")
        .merge(target_users, on="KaiinID", how="inner")
    )

    similar_aucs = (
        pd.concat([users_watch[users_watch["TourokuDate"] < period["oldest"]][["KaiinID", key]],
                   users_bid[users_bid["ShudouNyuusatsuDate"] < period["oldest"]][["KaiinID", key]]])
            .drop_duplicates()
            .merge(auction[auction["CreateDate"] < period["newest"]][["AuctionID", key]], on=key)[
            ["KaiinID", "AuctionID"]]
            .drop_duplicates()
    )

    return similar_aucs

