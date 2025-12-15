import pandas as pd
import numpy as np
from pathlib import Path
import random

# 输入 IXI CSV
MAIN_CSV = "/home/blue/Blue_Project/IXI_full_IDdir.csv"

# 输出目录
OUT_DIR = Path("/home/blue/Blue_Project/IXI_train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件路径
TRAIN_CSV = OUT_DIR / "train_split.csv"
VAL_CSV   = OUT_DIR / "val_split.csv"
TEST_CSV  = OUT_DIR / "test_split.csv"

# 固定随机种子
random.seed(42)

# 读取总表
df = pd.read_csv(MAIN_CSV)
df = df[pd.to_numeric(df["Age"], errors="coerce").notnull()]
df["Age"] = df["Age"].astype(float)

# ⭐ IXI 没有 site 信息，这里按病人随机划分
patient_ids = sorted(df["PatientID"].unique().tolist())
random.shuffle(patient_ids)

num_total = len(patient_ids)
num_train = int(num_total * 0.6)
num_val   = int(num_total * 0.2)
num_test  = num_total - num_train - num_val

train_ids = patient_ids[:num_train]
val_ids   = patient_ids[num_train:num_train+num_val]
test_ids  = patient_ids[num_train+num_val:]

# 根据 PatientID 分割数据
train_df = df[df["PatientID"].isin(train_ids)]
val_df   = df[df["PatientID"].isin(val_ids)]
test_df  = df[df["PatientID"].isin(test_ids)]

# 保存
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n=== DONE ===")
print(f"Train: {len(train_df)} → {TRAIN_CSV}")
print(f"Val:   {len(val_df)} → {VAL_CSV}")
print(f"Test:  {len(test_df)} → {TEST_CSV}")
