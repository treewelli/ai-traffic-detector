import pandas as pd
import os
import cudf
import glob

# 替换成你的原始 CSV 文件路径
csv_path = os.path.join(os.path.dirname(__file__), "..", "data/raw", "TFTP.csv")

df = cudf.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace('[^a-zA-Z0-9]', '_', regex=True).str.lower()
# 查看标签的唯一值
unique_labels = df['label'].unique()
print("标签字段包含的类别：", unique_labels)

# 统计每个标签出现的次数
label_counts = df['label'].value_counts()
print("标签分布：\n", label_counts)
