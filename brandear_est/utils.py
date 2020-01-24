import re
import datetime

import pandas as pd
import numpy as np


def to_datetime(df):
    cols = df.columns
    datestr_cols = [col for col in cols if re.search('[Dd]ate$', col)]
    for datestr_col in datestr_cols:
        df[datestr_col] = pd.to_datetime(df[datestr_col], format='%Y-%m-%d %H:%M:%S')
    return df


def to_pickle(obj, dirname, filename):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    print(filename + " : " + now)
    obj.to_pickle(dirname + now + "_" + filename)


def read_csv(path_name):
    df = reduce_mem_usage(pd.read_csv(path_name))
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def drop(df, cols):

    drop_cols = [col for col in cols if col in df.columns]
    df_droped = df.drop(drop_cols, axis=1)
    return df_droped


def left_anti_join(left_df, right_df, left_key, right_key):
    """二つのDataFrameを指定のキーで結合し、左側のDataFrameのうち、右側のDataFrameにキーが存在しないレコードのみを返す関数

    Parameters
    ----------
    left_df : DataFrame
        左側のDataFrame
    right_df : DataFrame
        右側のDataFrame
    left_key : str or List[str]
        左側の結合キーのカラム名（またはそのリスト）
    right_key : str or List[str]
        左側の結合キーのカラム名（またはそのリスト）

    Returns
    -------
    difference_df : DataFrame

    """
    left_anti_df = left_df.merge(
        right=right_df[right_key],
        how="left",
        left_on=left_key,
        right_on=right_key,
        indicator=True
    )[lambda x: x._merge == "left_only"].drop_duplicates().drop('_merge', 1)[left_df.columns]
    return left_anti_df


def cross_join(df1, df2):
    df1["cross_flg"] = 0
    df2["cross_flg"] = 0
    cross_df = df1.merge(df2, on=["cross_flg"], how="outer").drop("cross_flg", axis=1)
    return cross_df