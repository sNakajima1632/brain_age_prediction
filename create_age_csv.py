import csv
import re
from pathlib import Path
import pandas as pd

# ------------ PATH CONFIG ------------
PREPROC_ROOT = Path("/home/blue/Blue_Project/CoRR_fully_Preprocessed")
BIDS_ROOT    = Path("/home/blue/Blue_Project/CoRR/RawDataBIDS")
OUT_CSV      = Path("/home/blue/Blue_Project/CoRR_MNI_with_Age.csv")

# ------------ AGE COLUMN DETECTION ------------
age_candidates = [
    "age_at_scan_1", "age", "Age", "AGE_YRS", "age_years",
    "Age_in_Yrs", "age_year"
]

def detect_age_column(df):
    """Return the first column that looks like an age column."""
    for col in df.columns:
        if col in age_candidates:
            return col
    # fallback: any numeric column
    for col in df.columns:
        try:
            if pd.to_numeric(df[col], errors="coerce").notnull().any():
                return col
        except:
            continue
    return None


def load_participants(tsv_path):
    """
    Read participants.tsv robustly:
    - Handle broken header
    - Auto-detect ID column (first column)
    - Auto-detect age column
    """
    try:
        df = pd.read_csv(tsv_path, sep="\t", engine="python")
    except:
        df = pd.read_csv(tsv_path, sep="\s+", engine="python")

    df = df.dropna(how="all")  # drop empty rows
    df.columns = df.columns.map(str.strip)

    # ID column = ALWAYS first column
    id_col = df.columns[0]

    # Detect age column
    age_col = detect_age_column(df)
    if age_col is None:
        return None, None, None

    return df, id_col, age_col


def main():
    id_to_age = {}

    # ------------ BUILD GLOBAL ID->AGE MAP ------------
    print("Scanning all participants.tsv ...")
    for tsv in BIDS_ROOT.rglob("participants.tsv"):
        df, id_col, age_col = load_participants(tsv)
        if df is None:
            print(f"[WARN] Could not parse {tsv}")
            continue

        for _, row in df.iterrows():
            raw_id = str(row[id_col]).strip()
            m = re.search(r"(\d+)", raw_id)
            if not m:
                continue
            numeric_id = m.group(1)

            try:
                age = float(row[age_col])
            except:
                continue

            id_to_age[numeric_id] = age

    print(f"Total subjects with age info: {len(id_to_age)}")

    # ------------ MATCH WITH PREPROCESSED DATA ------------
    rows = [("PatientID", "Age", "Path")]
    matched = 0
    missed = 0

    for folder in sorted(PREPROC_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        pid = folder.name
        m = re.search(r"(\d+)", pid)
        if not m:
            missed += 1
            continue
        numeric_id = m.group(1)

        mni_path = folder / f"{pid}_T1w_MNI.nii.gz"
        if not mni_path.exists():
            missed += 1
            continue

        if numeric_id not in id_to_age:
            # Debug print
            print(f"[MISS] No age for {pid} (ID={numeric_id})")
            missed += 1
            continue

        age = id_to_age[numeric_id]
        rows.append((pid, age, str(mni_path)))
        matched += 1

    # ------------ SAVE CSV ------------
    with OUT_CSV.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"\n✔ CSV generated: {OUT_CSV}")
    print("Matched subjects:", matched)
    print("No match:", missed)


if __name__ == "__main__":
    main()
