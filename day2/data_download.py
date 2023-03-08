import os
import gzip
from time import sleep
from urllib import request

import pandas as pd
from datetime import datetime, date, timedelta

# helper function
def download(url, filepath):
    request.urlretrieve(url, filepath)
    df = unzip(filepath)
    os.remove(filepath)
    return df

#open zip
def unzip(filepath):
    with gzip.open(filepath, "rt") as f:
        df = pd.read_csv(f)
    return df


symbol = "BTCUSD"
baseurl = f'https://public.bybit.com/trading/{symbol}/'
# データの開始日時，終了日時
start, end = date(2022, 3, 1), date(2022, 3, 31)

exec_data = pd.DataFrame()
for i in range((end -start).days + 1):
    date_str = start + timedelta(i)
    date_Str = date_str.strftime("%Y-%m-%d")
    filepath = f"{date_str}.csv"
    dlurl = baseurl + f"{symbol}{date_str}.csv.gz"
    # urlを指定してzip形式でダウンロード→開封しpandasでDataFrame形式に変換
    df2 = download(dlurl, filepath)
    exec_data = pd.concat([exec_data, df2])
    print(f"downloaded {date_str}")
    sleep(0.5)

# ダウンロードしたデータを確認
exec_data.head()

# ダウンロードしたデータをpickle形式で保存
pickle_file_name = f"bybit_{symbol}_2022_3.pkl"
exec_data.to_pickle(pickle_file_name)