import os
import math
import gzip
from time import sleep
from random import gauss
from itertools import product
import pickle
from datetime import datetime, timedelta, date
from urllib import request

import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# pickle dataの読み込み
with open("./bybit_BTCUSD_2022_3.pkl", mode="rb") as f:
    exec_data = pickle.load(f)
    # unix timestampを日付に変換
    exec_data["timestamp"] = pd.to_datetime(exec_data["timestamp"], unit = "s")
    exec_data.set_index("timestamp", inplace=True)
    exec_data.head()

# dataの概要
print(exec_data.head())
print(len(exec_data))

# 非構造化データを構造化する(約定履歴→バー形式)
# サンプリング期間
FREQ = "15min"
time_bar = pd.DataFrame(columns=["op", "hi", "lo", "cl", "volume"])
time_bar.index.name = "timestamp"

ohlc = exec_data["price"].resample(FREQ).ohlc()
volume = exec_data["size"].resample(FREQ).sum()
ohlcv = pd.concat([ohlc, volume], axis = 1)
ohlcv.index.name = "timestamp"
ohlcv.columns = ["op", "hi", "lo", "cl", "volume"]

time_bar = pd.concat([time_bar, ohlcv], axis = 0)
print(time_bar.head())

# メモリ節約のため約定データ削除
del exec_data

# 定常性のあるデータに変換
df = time_bar.copy()
# 対数処理
df["log_open"] = np.log(df["op"])
df["log_high"] = np.log(df["hi"])
df["log_low"] = np.log(df["lo"])
df["log_close"] = np.log(df["cl"])
# 対数差分処理
df["diff_log_open"] = df["log_open"].diff()
df["diff_log_high"] = df["log_high"].diff()
df["diff_log_low"] = df["log_low"].diff()
df["diff_log_close"] = df["log_close"].diff()
df["diff_log_volume"] = df["volume"].diff()
df.dropna(inplace=True)

# helper function
# train validationのための分割関数
def timeseries_train_val_split(Xy, target="y"):
    # 時系列の前半75％を学習, 後半25％を検証に利用
    train, val = Xy[:int(len(Xy) * 0.75)], Xy[int(len(Xy) * 0.75)+10:]
    trainX = train.drop([target], axis = 1)
    trainy = train[target]
    valX = val.drop([target], axis = 1)
    valy = val[target]
    return trainX, trainy, valX, valy

from sklearn.metrics import accuracy_score
# 上昇or下落の予測精度を測定する関数
def eval_direction(target, pred):
    target = np.where(np.array(target) > 0, 1, -1)
    pred = np.where(np.array(pred) > 0, 1, -1)
    print("accuracy", accuracy_score(target, pred))

# ARモデルを利用したい
# データyが定常である必要がある
# データyの定常性をADF検定で確認
from statsmodels.tsa.stattools import adfuller
def adf_test(data, sig_level = 0.05, do_print = True) -> bool:
    """
    ADF検定を実施する関数
    Args:
        data: 検定対象の系列データ
        sig_level: 有意水準
        do_print: 検定結果をprintするかどうか
    Returns:
        bool: Trueの場合定常,Falseの場合非定常を表す
    """
    if do_print:
        print("results of Dickey-Fuller Examination: ")
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:4], index = ["Test Static", "p-value", "#Lags Used", "Number of Observations Used"])
    if do_print:
        print(dfoutput)
    return dfoutput["p-value"] < sig_level

# 定常性の確認
# time_barの場合
if adf_test(df["log_close"], do_print=True):
    print("対数価格系列は定常")
else:
    print("対数価格系列は非定常")
# 差分をとったtime_barの場合
if adf_test(df["diff_log_close"], do_print=True):
    print("1次差分系列は定常である")
else:
    print("1次差分系列は非定常である")


# AR(1)モデルを用いた分析
from sklearn.linear_model import LinearRegression
# データと予測対象になるラベルの用意
Xy = df[["diff_log_close"]]
# 予測対象: diff_log_closeの1ステップ先の値
Xy["y"] = df["diff_log_close"].shift(-1)
Xy.dropna(inplace = True)
print(Xy.head())

# trainとvalidationの分割
trainX, trainy, valX, valy = timeseries_train_val_split(Xy, target = "y")

# AR(1)モデルの設定とfitting
lr = LinearRegression()
lr.fit(trainX[["diff_log_close"]], trainy)
pred = lr.predict(valX[["diff_log_close"]])

# 可視化

# 評価
print(eval_direction(valy, pred))

# 機械学習モデル
# 一般にはテーブル形式のデータに適用
# そのため各行は前後の行との相関を持たないように定常化する
# データとラベルの用意
Xy = df[["diff_log_open", "diff_log_high", "diff_log_low", "diff_log_close", "diff_log_volume"]]
Xy["y"] = df["diff_log_close"].shift(-1)
Xy.dropna(inplace=True)
print(Xy.head())

# random forest modelの用意
from sklearn.ensemble import RandomForestRegressor
#trainとvalidationを分割
trainX, trainy, valX, valy = timeseries_train_val_split(Xy, target = "y")

# modelの定義とfitting
rf = RandomForestRegressor(random_state = 0)
rf.fit(trainX, trainy)
pred = rf.predict(valX)

# 可視化

# 評価
print(eval_direction(valy, pred))


# deep learning
# dlは必ずしも定常化処理が必要ではない
# 学習安定のために今回は標準化処理を実施
# 未来の情報を用いないように，直近10本文のデータの平均と分散を用いて処理
# データとラベルの用意

# 過去10本分の平均と標準偏差を用いて標準化を実施(未来情報を参照しないように標準化)
df["scaled_log_open"] = (df["log_open"] - df["log_open"].rolling(10).mean()) / df["log_open"].rolling(10).std()
df["scaled_log_high"] = (df["log_high"] - df["log_high"].rolling(10).mean()) / df["log_high"].rolling(10).std()
df["scaled_log_low"] = (df["log_low"] - df["log_low"].rolling(10).mean()) / df["log_low"].rolling(10).std()
df["scaled_log_close"] = (df["log_close"] - df["log_close"].rolling(10).mean()) / df["log_close"].rolling(10).std()

Xy = df[["scaled_log_open", "scaled_log_high", "scaled_log_low", "scaled_log_close", "diff_log_close"]]
Xy["y"] = df["diff_log_close"].shift(-1)
Xy.dropna(inplace=True)

# LSTMの学習の用意
from torch import nn
# LSTMを用いた回帰モデル
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first):
        super(LSTMRegressor, self).__init__()
        # lstm層の設定
        self.lstm = nn.LSTM(
            input_size = input_size, hidden_size = hidden_size, batch_first = batch_first
        )
        # 出力層の設定
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, inouts):
        h, _ = self.lstm(inputs)
        # LSTM層から出力される隠れ層を出力層にとおして予測結果を得る
        output = self.output_layer(h[:, -1])
        return output