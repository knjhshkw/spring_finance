import numpy as np
import numpy.random as random

# 小数第三位まで表示
# %precision 3

# array
data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
print(data)
print("type", data.dtype)
print("dim", data.ndim)
print("size", data.size)

data.sort()
print("after sort", data)

# 降順
data[::-1].sort()
print("after sort", data)

# seed
random.seed(0)
# 正規分布の乱数を10個生成
rnd_data = random.randn(10)
print(rnd_data)

# 抽出対象データ
data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
print(random.choice(data, 10))
print("重複なし", random.choice(data, 10, replace=False))

# matrix
array1 = np.arange(9).reshape(3, 3) # 0-8までの整数を3*3行列に変換
print(array1)

array2 = np.arange(9, 18).reshape(3, 3)
print(array2)
print(np.dot(array1, array2))

print(np.zeros((2, 3), dtype=np.int64))
print(np.ones((2, 3), dtype=np.float64))

# copyして別objectを作成
sample_array = np.arange(10)
sample_array_copy = np.copy(sample_array)
print(sample_array_copy)
sample_array_copy[0:3] = 20
print(sample_array_copy)

# pool index参照
sample_names = np.array(["a", "b", "c", "d", "a"])

random.seed(0)
data = random.randn(5, 5)

print(sample_names == "a")
print(data[sample_names == "a"])    # 0番目と4番目の列が取り出される

# 条件制御のためのプール配列を作成
cond_data = np.array([True, True, False, False, True])
x_array = np.array([1, 2, 3, 4, 5])
y_array = np.array([100, 200, 300, 400, 500])
print(np.where(cond_data, x_array, y_array))

# 重複削除
cond_data = np.array([True, True, False, False, True])
print(np.unique(cond_data))

# 真偽値の判定
cond_data = np.array([True, True, False, False, True])
print("Trueが少なくとも1つあるか", cond_data.any())
print("すべてTrueか", cond_data.all())

# 条件合致の個数
sample_multi_array_data1 = np.arange(9).reshape(3, 3)
print(sample_multi_array_data1)
print("５より大きい数字がいくつか", (sample_multi_array_data1>5).sum())

# 行列計算
print("対角成分", np.diag(sample_multi_array_data1))
print("対角成分の和", np.trace(sample_multi_array_data1))
sample_multi_array_data2 = np.arange(16).reshape(4, 4)
print(sample_multi_array_data2)
print("すべての要素の平方根行列", np.sqrt(sample_multi_array_data2))

# 行列の次元変更(再形成)
sample_array = np.arange(10)
sample_array2 = sample_array.reshape(2, 5)
print(sample_array2)

# データの結合
sample_array3 = np.array([[1, 2, 3], [4, 5, 6]])
sampple_array4 = np.array([[7, 8, 9], [10, 11, 12]])
# 行方向に結合
print(np.concatenate([sample_array3, sampple_array4], axis = 0))
# vstackを使った行方向結合
print(np.vstack((sample_array3, sampple_array4)))

# 配列の分割
sample_array_vstack = np.vstack((sample_array3, sampple_array4))
first, second, third = np.split(sample_array_vstack, [1, 3])
print(first)

# 繰り返し処理
print(first.repeat(5))

# ブロードキャスト(配列の大きさが異なるときに自動で揃える機能)
sample_array = np.arange(10)
print(sample_array + 3)