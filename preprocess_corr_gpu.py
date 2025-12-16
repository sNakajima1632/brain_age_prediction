import os
import torch
import pandas as pd
from pathlib import Path
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRangePercentiles,
    Resize,
)

# ---------------------------
# GPU 设置
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# MONAI 预处理步骤（全部 GPU 加速）
# ---------------------------
preprocess_transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0)),
    ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
    ),
    Resize((160, 192, 160)),
])

# ---------------------------
# 单个受试者预处理
# ---------------------------
def process_subject(row, output_root):
    subject_id = row["PatientID"]
    t1_path = row["T1"]
    
    subj_out_dir = output_root / subject_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUNNING] {subject_id}")

    # Step 1. MONAI transforms（纯 GPU）
    data = preprocess_transform(t1_path)
    data = torch.tensor(data, dtype=torch.float32).to(device)

    # Step 2. 保存 .pt（训练用）
    pt_path = subj_out_dir / f"{subject_id}.pt"
    torch.save(data.cpu(), pt_path)

    # Step 3. 保存预处理后 NIfTI（可视化）
    nii_path = subj_out_dir / f"{subject_id}_preprocessed.nii.gz"
    img_np = data.cpu().numpy()[0]
    nii_img = nib.Nifti1Image(img_np, affine=np.eye(4))
    nib.save(nii_img, str(nii_path))

    print(f"[DONE] {subject_id}\n")

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    csv_file = "/CoRR_Preprocessed_csv.csv"
    output_dir = Path("/CoRR_Preprocessed")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} subjects")

    for idx, row in df.iterrows():
        print(f"=== ({idx+1}/{len(df)}) {row['PatientID']} ===")
        try:
            process_subject(row, output_dir)
        except Exception as e:
            print(f"[ERROR] {row['PatientID']}: {e}")
