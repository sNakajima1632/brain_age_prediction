import pandas as pd
from pathlib import Path


def create_corr_csv(corr_root, output_csv):
    """
    Generate CSV for CoRR dataset - T1 only version
    """
    corr_path = Path(corr_root)
    records = []

    # Iterate through all subject folders
    for sub_dir in sorted(corr_path.glob('sub-*')):
        subject_id = sub_dir.name
        anat_dir = sub_dir / 'anat'

        if not anat_dir.exists():
            print(f"Skipping {subject_id}: no anat directory")
            continue

        # Find T1 images
        t1_files = list(anat_dir.glob('*_T1w.nii.gz'))

        if not t1_files:
            print(f"Skipping {subject_id}: no T1w found")
            continue

        # For T1-only processing, duplicate T1 path for T2 column
        # OR just use T1 column only
        records.append({
            'PatientID': subject_id,
            'T1': str(t1_files[0]),
            'T2': str(t1_files[0])  # Using same T1 as placeholder
        })

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nCreated CSV with {len(df)} subjects: {output_csv}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df


if __name__ == "__main__":
    df = create_corr_csv('CoRR_data', 'corr_subjects.csv')