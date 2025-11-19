import pandas as pd
from pathlib import Path
import os
import re


def create_ixi_csv(ixi_t1_orig_root, ixi_t2_orig_root, xls_path, output_csv):
    """
    Generate CSV for IXI dataset including:
    - PatientID
    - Age (loaded from metadata file)
    - Original T1 path
    - Original T2 path
    """

    # Load metadata from IXI.xls
    try:
        meta_df = pd.read_excel(xls_path)
    except Exception:
        meta_df = pd.read_csv(xls_path)

    # Find a column containing 'id' (case-insensitive) instead of requiring exact 'IXI_ID'
    id_col = next((c for c in meta_df.columns if 'id' in c.lower()), None)
    age_col = next((c for c in meta_df.columns if 'age' in c.lower()), None)
    if id_col is None or age_col is None:
        raise ValueError(f"{xls_path} must contain an ID column and AGE column")

    # Convert the detected ID column to string for matching folder names
    meta_df[id_col] = meta_df[id_col].astype(str)

    # Create lookup dictionary using the detected ID column
    age_lookup = dict(zip(meta_df[id_col], meta_df[age_col]))

    # Build filename -> subject ID maps for T1 and T2 using a regex like in update_csv()
    t1_root = Path(ixi_t1_orig_root)
    t2_root = Path(ixi_t2_orig_root)

    pattern = re.compile(r"IXI(\d+)", re.IGNORECASE)
    file_map_t1 = {}
    file_map_t2 = {}

    if t1_root.exists():
        for img in os.listdir(t1_root):
            if not img.lower().endswith(('.nii', '.nii.gz')):
                continue
            m = pattern.search(img)
            if m:
                file_map_t1[m.group(1)] = str(t1_root / img)

    if t2_root.exists():
        for img in os.listdir(t2_root):
            if not img.lower().endswith(('.nii', '.nii.gz')):
                continue
            m = pattern.search(img)
            if m:
                file_map_t2[m.group(1)] = str(t2_root / img)

    # Iterate metadata and build records by matching numeric ID extracted from the metadata ID column
    records = []
    id_digits_re = re.compile(r"(\d+)")
    for _, row in meta_df.iterrows():
        raw_id = str(row[id_col])
        m = id_digits_re.search(raw_id)
        if not m:
            print(f"Skipping {raw_id}: cannot extract numeric ID")
            continue
        id_digits = m.group(1)

        # Only include if both T1 and T2 exist in the provided directories
        t1_path = file_map_t1.get(id_digits)
        t2_path = file_map_t2.get(id_digits)
        if not t1_path or not t2_path:
            print(f"Skipping {raw_id}: missing T1 or T2 in provided roots")
            continue

        records.append({
            'PatientID': raw_id,
            'Age': age_lookup.get(raw_id, ''),
            'T1_orig': t1_path,
            'T2_orig': t2_path,
        })

    # Save output CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\nCreated CSV with {len(df)} subjects: {output_csv}")
    print("\nFirst few rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    df = create_ixi_csv(
        ixi_t1_orig_root='IXI-T1',
        ixi_t2_orig_root='IXI-T2',
        xls_path='IXI.xls',
        output_csv='ixi_subjects.csv'
    )
