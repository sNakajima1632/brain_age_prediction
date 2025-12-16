import pandas as pd
import numpy as np

# 读取两个CSV文件
corr_df = pd.read_csv('/CoRR_MNI_with_Age.csv')
ixi_df = pd.read_csv('/IXI_full.csv')

print("=== 原始数据统计 ===")
print(f"CoRR原始数据: {len(corr_df)} 条")
print(f"IXI原始数据: {len(ixi_df)} 条")

# 从IXI数据中只保留PatientID, Age和T1列
ixi_filtered = ixi_df[['PatientID', 'Age', 'T1']].copy()
ixi_filtered.rename(columns={'T1': 'Path'}, inplace=True)

# 合并两个数据框
combined_df = pd.concat([corr_df, ixi_filtered], ignore_index=True)

print(f"\n合并后数据: {len(combined_df)} 条")

# 数据清洗
print("\n=== 开始清洗数据 ===")

# 1. 替换特殊字符（#等）为NaN
combined_df['Age'] = combined_df['Age'].replace('#', np.nan)
combined_df['Age'] = combined_df['Age'].replace('', np.nan)

# 2. 转换Age为数值类型
combined_df['Age'] = pd.to_numeric(combined_df['Age'], errors='coerce')

# 3. 删除Age为空的行
before_drop = len(combined_df)
combined_df = combined_df.dropna(subset=['Age'])
print(f"删除了 {before_drop - len(combined_df)} 条Age缺失的记录")

# 4. 删除Path为空的行
before_drop = len(combined_df)
combined_df = combined_df.dropna(subset=['Path'])
print(f"删除了 {before_drop - len(combined_df)} 条Path缺失的记录")

# 5. 可选：删除年龄异常值（例如<0或>120）
before_drop = len(combined_df)
combined_df = combined_df[(combined_df['Age'] >= 0) & (combined_df['Age'] <= 120)]
print(f"删除了 {before_drop - len(combined_df)} 条年龄异常的记录")

# 6. 重置索引
combined_df = combined_df.reset_index(drop=True)

# 保存清洗后的CSV
output_path = '/Combined_CoRR_IXI_cleaned.csv'
combined_df.to_csv(output_path, index=False)

print(f"\n=== 清洗完成 ===")
print(f"最终数据: {len(combined_df)} 条")
print(f"保存路径: {output_path}")

# 显示统计信息
print("\n=== 年龄统计 ===")
print(combined_df['Age'].describe())
print(f"\n年龄范围: {combined_df['Age'].min():.1f} - {combined_df['Age'].max():.1f}")

# 显示数据来源分布（如果PatientID有前缀可以区分）
print("\n=== 数据预览 ===")
print(combined_df.head(10))

# 检查是否有重复的PatientID
duplicates = combined_df['PatientID'].duplicated().sum()
print(f"\n重复的PatientID数量: {duplicates}")
if duplicates > 0:
    print("建议检查重复数据！")