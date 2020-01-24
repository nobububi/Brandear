import numpy as np
import pandas as pd
import lightgbm as lgb

from .utils import drop


class LgbBinaryClassifier():
    def __init__(self, ):
        self.watch_valid_model = None
        self.watch_sub_model = None
        self.bid_valid_model = None
        self.bid_sub_model = None

        self.params = {
            "objective": "binary",
            'metric': 'auc',
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
                    .sample(n=actioned_data.shape[0] * 10),
                actioned_data
            ])
        )
        return sampled_data

    def adjust_data(self, data, label):
        data_copy = data.copy()
        drop_cols = ["KaiinID", "AuctionID", "watch_actioned", "bid_actioned",
                     "CreateDate", "watch_ua_cnt", "watch_ua_newest", "watch_ua_oldest", "watch_period",
                     "bid_ua_cnt", "bid_ua_newest", "bid_ua_oldest", "bid_period"]

        lgb_dataset = lgb.Dataset(
            data=np.array(drop(data, drop_cols)),
            label=np.array(data[label])
        )

        return lgb_dataset

    def train(self, train_data, valid_data):

        sampled_train_data = self.sample_nonactioed(train_data)

        lgb_watch_train_set = self.adjust_data(sampled_train_data, "watch_actioned")
        lgb_watch_valid_set = self.adjust_data(valid_data, "watch_actioned")
        self.watch_valid_model = lgb.train(
            params=self.params,
            train_set=lgb_watch_train_set,
            valid_sets=lgb_watch_valid_set
        )

        lgb_bid_train_set = self.adjust_data(sampled_train_data, "bid_actioned")
        lgb_bid_valid_set = self.adjust_data(valid_data, "bid_actioned")
        self.bid_valid_model = lgb.train(
            params=self.params,
            train_set=lgb_bid_train_set,
            valid_sets=lgb_bid_valid_set
        )

    def retrain(self, train_data):

        sampled_train_data = self.sample_nonactioed(train_data)
        lgb_watch_train_set = self.adjust_data(sampled_train_data, "watch_actioned")
        self.watch_sub_model = lgb.train(
            params=self.params,
            train_set=lgb_watch_train_set
        )
        lgb_bid_train_set = self.adjust_data(sampled_train_data, "bid_actioned")
        self.bid_sub_model = lgb.train(
            params=self.params,
            train_set=lgb_bid_train_set
        )

    def predict(self, data):
        drop_cols = ["KaiinID", "AuctionID", "watch_actioned", "bid_actioned",
                     "CreateDate", "watch_ua_cnt", "watch_ua_newest", "watch_ua_oldest", "watch_period",
                     "bid_ua_cnt", "bid_ua_newest", "bid_ua_oldest", "bid_period"]
        watch_pred = (
            self.watch_sub_model.predict(np.array(drop(data, drop_cols)))
        )
        bid_pred = (
            self.bid_sub_model.predict(np.array(drop(data, drop_cols)))
        )
        pred = data[["KaiinID", "AuctionID"]].copy()
        pred["score"] = watch_pred * 0.2 + bid_pred * 0.8

        return pred