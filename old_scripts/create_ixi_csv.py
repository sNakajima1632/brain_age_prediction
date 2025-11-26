import pandas as pd
from pathlib import Path


def create_ixi_csv(ixi_root, xls_path, output_csv):
    """
    Generate CSV for IXI dataset including:
    - PatientID
    - Preprocessed T1 path
    - Preprocessed T2 path
    - Age (loaded from IXI.xls)
    """

    # Load metadata from IXI.xls
    try:
        meta_df = pd.read_excel(xls_path)
    except Exception:
        meta_df = pd.read_csv(xls_path)

    if "IXI_ID" not in meta_df.columns or "AGE" not in meta_df.columns:
        raise ValueError("IXI.xls must contain columns: IXI_ID, AGE")

    # Convert IXI_ID to string for matching folder names
    meta_df["IXI_ID"] = meta_df["IXI_ID"].astype(str)

    # Create lookup dictionary
    age_lookup = dict(zip(meta_df["IXI_ID"], meta_df["AGE"]))

    # Scan preprocessed IXI directory
    ixi_path = Path(ixi_root)
    records = []

    for sub_dir in sorted(ixi_path.glob('*')):
        subject_id = sub_dir.name  # folder name e.g., "2", "450"

        # Must match IXI_ID in metadata
        if subject_id not in age_lookup:
            print(f"Skipping {subject_id}: Not found in IXI.xls")
            continue

        # Find T1
        t1_files = list(sub_dir.glob('*_t1.nii.gz'))
        if not t1_files:
            print(f"Skipping {subject_id}: no T1 found")
            continue

        # Find T2
        t2_files = list(sub_dir.glob('*_t2.nii.gz'))
        if not t2_files:
            print(f"Skipping {subject_id}: no T2 found")
            continue

        # Append record
        records.append({
            'PatientID': subject_id,
            'Age': age_lookup[subject_id],     # <-- ADDED
            'T1': str(t1_files[0]),
            'T2': str(t2_files[0])
        })

    # -------------------------------------------------------
    # 3. Save output CSV
    # -------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\nCreated CSV with {len(df)} subjects: {output_csv}")
    print("\nFirst few rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    df = create_ixi_csv(
        ixi_root='IXIprep_final_image_only',
        xls_path='IXI.xls',
        output_csv='ixi_out.csv'
    )
