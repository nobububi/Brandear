import numpy as np
import pandas as pd
import lightgbm as lgb

from .utils import drop


class LgbLambdaLank():
    def __init__(self, ):
        self.valid_model = None
        self.sub_model = None
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            "ndcg_at": 20,
            "nround": 500,
            "learning_rate": 0.01,
            "max_depth": 6,
            "num_leaves": 127
        }

    def sample_nonactioed(self, data):
        actioned_data = data.query("(watch_actioned == 1) | (bid_actioned == 1)")
        sampled_data = (
            pd.concat([
                data.query("(watch_actioned == 0) & (bid_actioned == 0)")
                    .sample(n=actioned_data.shape[0] * 100),
                actioned_data
            ])
        )
        return sampled_data

    def adjust_data(self, data):
        data_copy = data.copy()
        drop_cols = ["KaiinID", "AuctionID", "watch_actioned", "bid_actioned",
                     "CreateDate", "watch_ua_cnt", "watch_ua_newest", "watch_ua_oldest", "watch_period",
                     "bid_ua_cnt", "bid_ua_newest", "bid_ua_oldest", "bid_period"]
        data_copy.sort_values(["KaiinID", "AuctionID"], inplace=True)

        if {"watch_actioned", "bid_actioned"} - set(data.columns) == set([]):
            label = np.array(data_copy[["watch_actioned", "bid_actioned"]].astype(int)).max(axis=1)
        else:
            label = None

        weight = (
            np.stack([
                np.array(data_copy["watch_actioned"].astype(int)),
                (np.array(data_copy["bid_actioned"]).astype(int) * 2),
                np.ones((data_copy.shape[0],))
            ], 1).max(axis=1)
        )

        group = (
            data_copy[["KaiinID", "AuctionID"]]
            .groupby("KaiinID", as_index=False)
            .count()
            .sort_values("KaiinID")["AuctionID"]
        )

        lgb_dataset = lgb.Dataset(
            data=np.array(drop(data_copy, drop_cols)),
            label=label,
            weight=weight,
            group=group
        )

        return lgb_dataset

    def train(self, train_data, valid_data):

        sampled_train_data = self.sample_nonactioed(train_data)
        lgb_train_set = self.adjust_data(sampled_train_data)
        lgb_valid_set = self.adjust_data(valid_data)
        self.valid_model = lgb.train(
            params=self.params,
            train_set=lgb_train_set,
            valid_sets=lgb_valid_set
        )

    def retrain(self, train_data):

        sampled_train_data = self.sample_nonactioed(train_data)
        lgb_train_set = self.adjust_data(sampled_train_data)
        self.sub_model = lgb.train(
            params=self.params,
            train_set=lgb_train_set
        )

    def predict(self, data):
        drop_cols = ["KaiinID", "AuctionID", "watch_actioned", "bid_actioned",
                     "CreateDate", "watch_ua_cnt", "watch_ua_newest", "watch_ua_oldest", "watch_period",
                     "bid_ua_cnt", "bid_ua_newest", "bid_ua_oldest", "bid_period"]
        data_copy = data.copy()
        sorted_data = data_copy.sort_values(["KaiinID", "AuctionID"])
        pred = self.sub_model.predict(
            data=np.array(drop(sorted_data, drop_cols)),
            group=np.array(
                sorted_data[["KaiinID", "AuctionID"]].groupby("KaiinID", as_index=False).count().sort_values("KaiinID")[
                    "AuctionID"])
        )
        sorted_data["score"] = pred
        return sorted_data[["KaiinID", "AuctionID", "score"]]