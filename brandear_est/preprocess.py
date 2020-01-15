from brandear_est.feature_engineering import *


def build_dataset(watch, bid, auction, dset_type, dset_to_period, target_users=None):
    oldest_dtime, newest_dtime = (
        dset_to_period[dset_type]["oldest"],
        dset_to_period[dset_type]["newest"]
    )

    # データセット作成の対象となるユーザー一覧
    if dset_type != "submission":
        watch_actioned = (
            watch[(watch["TourokuDate"] >= oldest_dtime) & (watch["TourokuDate"] < newest_dtime)]
        )
        bid_actioned = (
            bid[(bid["ShudouNyuusatsuDate"] >= oldest_dtime) & (bid["ShudouNyuusatsuDate"] < newest_dtime)]
        )
        target_users = (
            pd.concat([watch_actioned, bid_actioned], sort=False)[["KaiinID"]]
                .drop_duplicates()
        )

    # リークを防ぐため、特徴量、choiced_auc作成用のデータから正解データ抽出期間時のデータを削除する
    watch_train = watch[watch["TourokuDate"] < oldest_dtime]
    bid_train = bid[bid["ShudouNyuusatsuDate"] < oldest_dtime]

    # 予測対象とするオークションをルール,0次ベースのロジックで限定する
    dataset = merge_choiced_aucs(target_users, watch_train, bid_train, auction)

    # 特徴量付与
    dataset = add_features(dataset, watch_train, bid_train, oldest_dtime)

    # 正解データ付与
    if dset_type != "submission":
        watch_actioned.loc["watch_actioned"] = 1
        bid_actioned.loc["bid_actioned"] = 1
        dataset = (
            dataset
                .merge(watch_actioned[["KaiinID", "AuctionID", "watch_actioned"]], on=["KaiinID", "AuctionID"],
                       how="left")
                .merge(bid_actioned[["KaiinID", "AuctionID", "bid_actioned"]], on=["KaiinID", "AuctionID"], how="left")
                .fillna(0)
        )

    return dataset


def merge_choiced_aucs(target_users, watch, bid, auction):
    # choiced_auc付与部分
    # あるユーザーが現在までにアクションをした商品と同じ商品IDのオークションを抽出
    # これが関数内で作成する学習データの大元
    target_aucs = (
        pd.concat([watch[["KaiinID", "ShouhinID"]], bid[["KaiinID", "ShouhinID"]]])
            .drop_duplicates()
            .merge(auction[["AuctionID", "ShouhinID"]], on="ShouhinID")[["KaiinID", "AuctionID"]]
            .drop_duplicates()
    )

    auc_cols = (
        ['AuctionID', 'ShouhinShubetsuID', 'ShouhinID', 'SaishuppinKaisuu',
         'ConditionID', 'BrandID', 'GenreID', 'GenreGroupID', 'LineID',
         'DanjobetsuID']
    )

    # 今回の対象ユーザーに絞る
    target_data = (
        target_users
            .merge(target_aucs, on="KaiinID")
            .merge(auction[auc_cols], on="AuctionID")
    )

    return target_data
