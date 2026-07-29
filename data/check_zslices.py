'''
统计xxx/split/all_anno.csv文件中zmaX列和zmin列差值的最大值最小值，如果差值小于3,输出最小值对应的pid
'''
import pandas as pd

# 读取CSV文件
df = pd.read_csv('NLSTSeg/split/all_anno.csv')

# 计算zmax和zmin的差值
df['diff'] = df['zmax'] - df['zmin']+1  # [ZMIN,ZMAX]

# 统计最大值和最小值
max_diff = df['diff'].max()
min_diff = df['diff'].min()

print(f"zmax - zmin 差值统计:")
print(f"最大值: {max_diff}")
print(f"最小值: {min_diff}")

if min_diff < 3:
    min_rows = df[df['diff'] == min_diff]
    print(f"\n最小值对应的记录:")
    print(min_rows[['pid', 'zmin', 'zmax', 'diff']].to_string(index=False))