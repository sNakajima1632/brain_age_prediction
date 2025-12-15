import pandas as pd

# 读取两个CSV文件
corr_df = pd.read_csv('/home/blue/Blue_Project/CoRR_MNI_with_Age.csv')
ixi_df = pd.read_csv('/home/blue/Blue_Project/IXI_full_IDdir.csv')

# 从IXI数据中只保留PatientID, Age和T1列，并重命名T1为Path
ixi_filtered = ixi_df[['PatientID', 'Age', 'T1']].copy()
ixi_filtered.rename(columns={'T1': 'Path'}, inplace=True)

# 合并两个数据框
combined_df = pd.concat([corr_df, ixi_filtered], ignore_index=True)

# 保存合并后的CSV
output_path = '/home/blue/Blue_Project/Combined_CoRR_IXI.csv'
combined_df.to_csv(output_path, index=False)

print(f"合并完成！")
print(f"CoRR数据: {len(corr_df)} 条")
print(f"IXI数据: {len(ixi_filtered)} 条")
print(f"合并后总计: {len(combined_df)} 条")
print(f"保存路径: {output_path}")

# 显示前几行预览
print("\n合并后数据预览:")
print(combined_df.head(10))

# 显示基本统计信息
print("\n年龄统计:")
print(combined_df['Age'].describe())