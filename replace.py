import csv
import os
from pathlib import Path
import pandas as pd

# -------- Paths --------
OLD_CSV = "/home/blue/Blue_Project/CoRR_Age_Training_Table.csv"
PREPROC_ROOT = Path("/home/blue/Blue_Project/CoRR_fully_Preprocessed")
OUT_CSV = Path("/home/blue/Blue_Project/CoRR_MNI_with_Age.csv")

def main():

    df = pd.read_csv(OLD_CSV)
    print("Loaded previous CSV:", len(df))

    rows = [("PatientID", "Age", "Path")]
    matched, missed = 0, 0

    for _, row in df.iterrows():
        pid = row["PatientID"]
        age = row["Age"]

        folder = PREPROC_ROOT / pid
        nii_path = folder / f"{pid}_T1w_MNI.nii.gz"

        if not nii_path.exists():
            print(f"[MISS] No MNI file for {pid}")
            missed += 1
            continue

        rows.append((pid, age, str(nii_path)))
        matched += 1

    with OUT_CSV.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("\n✔ CSV generated:", OUT_CSV)
    print("Matched subjects:", matched)
    print("Missing subjects:", missed)

if __name__ == "__main__":
    main()
