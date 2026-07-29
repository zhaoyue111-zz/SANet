'''
统计/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/all_anno_3D.csv中只有单层label和两层label的病例数
'''

import pandas as pd

csv_path = r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/all_anno_3D.csv"

df = pd.read_csv(csv_path)

# 按 index 统计每个病例/结节有几层 label
layer_counts = df["index"].value_counts().sort_index()

# 只有 1 层 label 的病例 index
one_layer_index = layer_counts[layer_counts == 1].index.tolist()

# 只有 2 层 label 的病例 index
two_layer_index = layer_counts[layer_counts == 2].index.tolist()

# 总病例数
total_cases = layer_counts.shape[0]

print("总病例数:", total_cases)

print("\n只有 1 层 label 的病例数:", len(one_layer_index))
print("比例: {:.2f}%".format(len(one_layer_index) / total_cases * 100))
print("index:", one_layer_index)

print("\n只有 2 层 label 的病例数:", len(two_layer_index))
print("比例: {:.2f}%".format(len(two_layer_index) / total_cases * 100))
print("index:", two_layer_index)

# 查看完整层数分布
dist = layer_counts.value_counts().sort_index()
print("\n完整 label 层数分布:")
print(dist)

'''
只有 1 层 label 的病例数: 20
比例: 21.05%
index: [1, 4, 6, 12, 14, 17, 21, 26, 27, 31, 34, 36, 45, 46, 54, 57, 68, 81, 88, 95]

只有 2 层 label 的病例数: 24
比例: 25.26%
index: [2, 3, 8, 13, 15, 16, 20, 30, 33, 35, 38, 41, 42, 44, 47, 56, 65, 72, 73, 75, 84, 90, 91, 93]
'''