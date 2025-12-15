import pandas as pd
import numpy as np
from pathlib import Path
import random

# 输入主 CSV
MAIN_CSV = "/home/blue/Blue_Project/IXI_full_IDdir.csv"

# 输出目录
OUT_DIR = Path("/home/blue/Blue_Project/T1_T2")
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

print(f"Total samples loaded: {len(df)}")
print(f"Age range: {df['Age'].min():.2f} - {df['Age'].max():.2f} years")
print(f"Unique patients: {df['PatientID'].nunique()}")

# 按PatientID分割（确保同一患者的T1和T2在同一个集合中）
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
train_df = df[df["PatientID"].isin(train_ids)].copy()
val_df   = df[df["PatientID"].isin(val_ids)].copy()
test_df  = df[df["PatientID"].isin(test_ids)].copy()

# 为T1和T2创建单独的行（每个样本变成两行：一行T1，一行T2）
def expand_modalities(df_subset):
    """将每个样本扩展为T1和T2两行"""
    t1_rows = []
    t2_rows = []
    
    for _, row in df_subset.iterrows():
        # T1行
        t1_row = {
            'PatientID': row['PatientID'],
            'Age': row['Age'],
            'Path': row['T1'],  # 使用预处理后的T1路径
            'Modality': 'T1'
        }
        t1_rows.append(t1_row)
        
        # T2行
        t2_row = {
            'PatientID': row['PatientID'],
            'Age': row['Age'],
            'Path': row['T2'],  # 使用预处理后的T2路径
            'Modality': 'T2'
        }
        t2_rows.append(t2_row)
    
    # 合并T1和T2
    expanded_df = pd.DataFrame(t1_rows + t2_rows)
    return expanded_df

# 扩展每个数据集
train_expanded = expand_modalities(train_df)
val_expanded = expand_modalities(val_df)
test_expanded = expand_modalities(test_df)

# 保存
train_expanded.to_csv(TRAIN_CSV, index=False)
val_expanded.to_csv(VAL_CSV, index=False)
test_expanded.to_csv(TEST_CSV, index=False)

print("\n=== DONE ===")
print(f"Train: {len(train_expanded)} samples ({len(train_ids)} patients, {len(train_expanded)//2} T1+T2 pairs)")
print(f"  → {TRAIN_CSV}")
print(f"Val:   {len(val_expanded)} samples ({len(val_ids)} patients, {len(val_expanded)//2} T1+T2 pairs)")
print(f"  → {VAL_CSV}")
print(f"Test:  {len(test_expanded)} samples ({len(test_ids)} patients, {len(test_expanded)//2} T1+T2 pairs)")
print(f"  → {TEST_CSV}")