import argparse
import logging
import os
import re
import pandas as pd
from pathlib import Path


def _extract_id(s: str) -> str:
    m = re.search(r"IXI(\d+)", str(s), re.IGNORECASE)
    return m.group(1) if m else ''


def create_ixi_csv(ixi_t1_root: str, ixi_t2_root: str, xls_path: str, output_csv: str) -> pd.DataFrame:
    """
    Create a CSV linking IXI subjects to T1/T2 image files and ages.

    Parameters:
    -----------
    ixi_t1_root: str
        directory containing T1 image files
    ixi_t2_root: str
        directory containing T2 image files
    xls_path: str
        path to IXI metadata (Excel or CSV)
    output_csv: str
        path to write output CSV

    The function attempts to match images using numeric IDs extracted from
    filenames (e.g. "IXI123") and from the metadata ID column.
    """

    logging.info("Loading metadata from %s", xls_path)
    try:
        meta_df = pd.read_excel(xls_path)
    except Exception:
        meta_df = pd.read_csv(xls_path)

    # Detect columns
    id_col = next((c for c in meta_df.columns if 'id' in c.lower()), None)
    age_col = next((c for c in meta_df.columns if 'age' in c.lower()), None)
    if id_col is None or age_col is None:
        raise ValueError(f"{xls_path} must contain an ID column and AGE column")

    # Normalize id column to int
    meta_df[id_col] = meta_df[id_col].astype(int)

    # Create age lookup by ID
    age_lookup_by_id = {}
    for _, row in meta_df.iterrows():
        raw_id = int(row[id_col])
        age = row[age_col]
        if raw_id:
            age_lookup_by_id[raw_id] = age

    # Map image files by extracted subject ID
    t1_root = Path(ixi_t1_root)
    t2_root = Path(ixi_t2_root)

    pattern = re.compile(r"IXI(\d+)", re.IGNORECASE)
    file_map_t1 = {}
    file_map_t2 = {}

    if t1_root.exists():
        for img in sorted(os.listdir(t1_root)):
            if not img.lower().endswith(('.nii', '.nii.gz')):
                continue
            m = pattern.search(img)
            if m:
                file_map_t1[int(m.group(1))] = str(t1_root / img)

    if t2_root.exists():
        for img in sorted(os.listdir(t2_root)):
            if not img.lower().endswith(('.nii', '.nii.gz')):
                continue
            m = pattern.search(img)
            if m:
                file_map_t2[int(m.group(1))] = str(t2_root / img)

    # Iterate through metadata and build records
    records = []
    for _, row in meta_df.iterrows():
        raw_id = int(row[id_col])

        t1_path = file_map_t1.get(raw_id)
        t2_path = file_map_t2.get(raw_id)
        age = age_lookup_by_id.get(raw_id)

        missing = []
        if not t1_path:
            missing.append("T1 image")
        if not t2_path:
            missing.append("T2 image")
        if age in [None, '', float('nan')] or pd.isna(age):
            missing.append("age")

        if missing:
            logging.warning("Skipping subject %s: missing %s", raw_id, ", ".join(missing))
            continue

        records.append({
            'PatientID': raw_id,
            'Age': age,
            'T1_orig': t1_path,
            'T2_orig': t2_path,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    logging.info("Created CSV with %d subjects: %s", len(df), output_csv)
    print("\nFirst few rows:")
    print(df.head())
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare IXI CSV linking subjects, ages and T1/T2 images')
    p.add_argument('--ixi_t1_root', default='IXI-T1', help='directory containing IXI T1 images')
    p.add_argument('--ixi_t2_root', default='IXI-T2', help='directory containing IXI T2 images')
    p.add_argument('--xls_path', default='IXI.xls', help='path to IXI metadata (Excel or CSV)')
    p.add_argument('--output_csv', default='ixi_subjects.csv', help='output CSV path')
    p.add_argument('--verbose', action='store_true', help='enable verbose logging')
    return p


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')

    create_ixi_csv(
        ixi_t1_root=args.ixi_t1_root,
        ixi_t2_root=args.ixi_t2_root,
        xls_path=args.xls_path,
        output_csv=args.output_csv,
    )
