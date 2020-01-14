import re

import pandas as pd


def to_datetime(df):
    cols = df.columns
    datestr_cols = [col for col in cols if re.search('[Dd]ate$', col)]
    for datestr_col in datestr_cols:
        df[datestr_col] = pd.to_datetime(df[datestr_col], format='%Y-%m-%d %H:%M:%S')
    return df

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