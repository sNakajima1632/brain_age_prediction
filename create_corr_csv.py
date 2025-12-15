from pathlib import Path
import pandas as pd


def create_corr_csv(corr_root: str, output_csv: str) -> pd.DataFrame:
    """
    扫描 CoRR/T1w 目录，生成包含所有 T1w 路径的 CSV。

    目录结构假设为：
    corr_root/
        SITE_1/
            sub-XXXX/
                ses-1/
                    anat/*_T1w.nii.gz
        SITE_2/
            ...

    CSV 字段：
        - PatientID: 组合的唯一 ID (SITE_subID_session)
        - T1: T1w 路径
        - T2: T1w 路径（保持与旧 ANTs 脚本兼容）
        - Site, SubID, Session
    """

    root = Path(corr_root)
    records = []

    for site_dir in sorted(root.iterdir()):
        if not site_dir.is_dir():
            continue
        site = site_dir.name

        # sub-XXXX
        for sub_dir in sorted(site_dir.glob("sub-*")):
            sub_id = sub_dir.name

            # session 可能存在
            session_dirs = list(sub_dir.glob("ses-*"))
            if session_dirs:
                session_candidates = session_dirs
            else:
                session_candidates = [sub_dir]  # 无 session 情况

            for ses_dir in session_candidates:
                ses = ses_dir.name if ses_dir != sub_dir else "NA"

                anat_dir = ses_dir / "anat"
                if not anat_dir.exists():
                    continue

                t1_files = sorted(anat_dir.glob("*_T1w.nii*"))
                if not t1_files:
                    continue

                t1_path = t1_files[0]

                # PatientID 用于兼容 ANTs 预处理脚本，同时保持全局唯一
                patient_id = f"{site}_{sub_id}_{ses}"

                records.append({
                    "PatientID": patient_id,
                    "T1": str(t1_path),
                    "T2": str(t1_path),   # 占位，不影响 T1-only pipeline
                    "Site": site,
                    "SubID": sub_id,
                    "Session": ses,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} subjects to {output_csv}")
    return df


if __name__ == "__main__":
    create_corr_csv(
        corr_root="/home/blue/Blue_Project/CoRR/T1w",
        output_csv="/home/blue/Blue_Project/CoRR_Preprocessed_csv.csv"
    )
