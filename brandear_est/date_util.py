from dateutil.relativedelta import relativedelta


def extract_recent_data(df, date_col, base_dtime, days):
    oldest_dtime = base_dtime - relativedelta(days=days)
    return df[df[date_col] > oldest_dtime]
