import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# Seriesの使い方(1次元配列のようなオブジェクト)
sample_pandas_data = pd.Series([0,10,20,30,40,50,60,70,80,90])
print(sample_pandas_data)
print("データの値", sample_pandas_data.values)
print("index", sample_pandas_data.index)

# indexをアルファベットで置換
sample_pandas_index_data = pd.Series(
    [0, 10,20,30,40,50,60,70,80,90],
    index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
)
print(sample_pandas_index_data)

# data frame
attri_data1 = {
    'ID':['100','101','102','103','104'],
    'City':['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
    'Birth_year':[1990,1989,1992,1997,1982],
    'Name':['Hiroshi','Akiko','Yuki','Satoru','Steve']
}
attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)
attri_data_frame_index1 = DataFrame(attri_data1, index=["a", "b", "c", "d", "e"])
print(attri_data_frame_index1)

# 行列操作
print("転置", attri_data_frame1.T)
print("列名指定抽出", attri_data_frame1.Birth_year)
print("複数列名", attri_data_frame1[["ID", "Birth_year"]])

# 条件フィルター
print(attri_data_frame1[attri_data_frame1["City"] == "Tokyo"])
print("複数条件フィルター", attri_data_frame1[attri_data_frame1["City"].isin(["Tokyo", "Osaka"])])

# データの列削除
attri_data_frame1.drop(["Birth_year"], axis=1)  # inplaceパラメータで元データ置き換え

# データの結合
attri_data2 = {
    'ID':['100','101','102','105','107'],
    'Math':[50,43,33,76,98],
    'English':[90,30,20,50,30],
    'Sex':['M','F','F','M','M']
}
attri_data_frame2 = DataFrame(attri_data2)
print(attri_data_frame2)
print(pd.merge(attri_data_frame1, attri_data_frame2))

# グループ集計
print(attri_data_frame2.groupby("Sex")["Math"].mean())

# ソート
attri_data2 = {
    'ID':['100','101','102','103','104'],
    'City':['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
    'Birth_year':[1990,1989,1992,1997,1982],
    'Name':['Hiroshi','Akiko','Yuki','Satoru','Steve']
}
attri_data_frame2 = DataFrame(attri_data2)
attri_data_frame_index2 = DataFrame(attri_data2,index=['e','b','a','d','c'])
attri_data_frame_index2
print("indexソート", attri_data_frame_index2.sort_index())
print("値ソート", attri_data_frame_index2.Birth_year.sort_values()) #ascending=Falseパラメータで降順