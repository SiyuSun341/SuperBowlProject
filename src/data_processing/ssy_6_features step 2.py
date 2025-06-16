import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# === set path ===
input_path = r".\video_feature_analysis_full.csv"
output_path = input_path.replace(".csv", "_encoded.csv")

# === try to read CSV, adapt to Chinese encoding ===
try:
    df = pd.read_csv(input_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding='gbk')  

# === automatically encode all object type fields (except video_id) ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == object and col != "video_id":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# === save encoded data ===
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"[✔] 编码完成，保存为：{output_path}")
