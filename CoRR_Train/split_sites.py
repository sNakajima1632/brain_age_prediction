import pandas as pd
import numpy as np
from pathlib import Path
import random

# 输入主 CSV
MAIN_CSV = "/home/blue/Blue_Project/CoRR_Age_Training_Table.csv"

# 输出目录
OUT_DIR = Path("/home/blue/Blue_Project/CoRR_Train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件路径
TRAIN_CSV = OUT_DIR / "train_split.csv"
VAL_CSV   = OUT_DIR / "val_split.csv"
TEST_CSV  = OUT_DIR / "test_split.csv"

# 固定随机种子以保证可重复
random.seed(42)

# 读取总表
df = pd.read_csv(MAIN_CSV)
df = df[pd.to_numeric(df["Age"], errors="coerce").notnull()]
df["Age"] = df["Age"].astype(float)

# 从 PatientID 提取 site（例如：BMB_1）
df["Site"] = df["PatientID"].apply(lambda x: x.split("_")[0])

# 获取所有独立 site
sites = sorted(df["Site"].unique().tolist())
print("\nAll Sites:", sites)
print("Total number of sites:", len(sites))

# 按 site 进行切分（60/20/20 比例）
num_sites = len(sites)
num_train = int(num_sites * 0.6)
num_val   = int(num_sites * 0.2)
# 剩下的为 test
num_test  = num_sites - num_train - num_val

random.shuffle(sites)

train_sites = sites[:num_train]
val_sites   = sites[num_train:num_train+num_val]
test_sites  = sites[num_train+num_val:]

print("\nTrain Sites:", train_sites)
print("Val Sites:", val_sites)
print("Test Sites:", test_sites)

# 根据 site 分割数据
train_df = df[df["Site"].isin(train_sites)]
val_df   = df[df["Site"].isin(val_sites)]
test_df  = df[df["Site"].isin(test_sites)]

# 保存
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n=== DONE ===")
print(f"Train: {len(train_df)} subjects → {TRAIN_CSV}")
print(f"Val:   {len(val_df)} subjects → {VAL_CSV}")
print(f"Test:  {len(test_df)} subjects → {TEST_CSV}")
