from pathlib import Path
import argparse
import logging
import re
import pandas as pd
from typing import Dict


"""
Create a combined IXI CSV with paths for original and preprocessed images.

- Load metadata (ID and age) from `IXI.xls` (Excel or CSV).
- Optionally scan `IXI_T1` and `IXI_T2` roots for original images and
  fill `T1_orig` and `T2_orig` columns.
- Optionally scan an `IXIprep` root (per-subject folders) for preprocessed
  images and fill `T1` and `T2` columns.
- Save a CSV with columns: `PatientID`, `Age`, `T1_orig`, `T2_orig`, `T1`, `T2`.

Usage examples:
    python IXIcsv_paths.py --xls_path IXI.xls --ixi_t1_root IXI-T1 --ixi_t2_root IXI-T2 --ixi_prep_root IXIprep --output_csv ixi_full.csv

If any of the three roots is omitted the corresponding columns will be left blank.
"""

def _read_metadata(xls_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(xls_path)
    except Exception:
        df = pd.read_csv(xls_path)
    return df


def _detect_cols(meta_df: pd.DataFrame):
    id_col = next((c for c in meta_df.columns if 'id' in c.lower()), None)
    age_col = next((c for c in meta_df.columns if 'age' in c.lower()), None)
    if id_col is None or age_col is None:
        raise ValueError('Metadata must contain an ID column and an AGE column')
    return id_col, age_col


def _build_file_maps_from_roots(root: Path, modality_tag: str = None, recursive: bool = False) -> Dict[str, str]:
    """
    Scan `root` for NIfTI files and map subject ids -> file path.

    The mapping keys include multiple variants so matching is robust:
    - the raw numeric group from filenames (may include leading zeros)
    - the integer form without leading zeros
    - the integer form zero-padded to 3 digits
    """
    file_map = {}
    if not root or not root.exists():
        return file_map

    pattern = re.compile(r"IXI(\d+)", re.IGNORECASE)
    globiter = root.rglob('*.nii*') if recursive else root.glob('*.nii*')
    for p in globiter:
        name = p.name
        m = pattern.search(name)
        if not m:
            # try to find any digits in filename as fallback
            m2 = re.search(r"(\d+)", name)
            if m2:
                g = m2.group(1)
            else:
                continue
        else:
            g = m.group(1)

        # create multiple keys
        try:
            ival = int(g)
            k_int = str(ival)
            k_padded = k_int.zfill(3)
        except Exception:
            k_int = g
            k_padded = g

        for key in (g, k_int, k_padded):
            if key not in file_map:
                file_map[key] = str(p)

    return file_map


def _build_prep_map_from_subject_folders(prep_root: Path) -> Dict[str, Dict[str, str]]:
    """
    Scan a preprocessed folder that contains per-subject directories.

    Returns a dict mapping subject_key -> {'t1': path, 't2': path}
    Subject_key will include multiple variants (raw, int, padded) similar
    to `_build_file_maps_from_roots`.
    """
    prep_map = {}
    if not prep_root or not prep_root.exists():
        return prep_map

    for sub in sorted(prep_root.iterdir()):
        if not sub.is_dir():
            continue
        subjname = sub.name
        try:
            ival = int(subjname)
            k_int = str(ival)
            k_padded = k_int.zfill(3)
        except Exception:
            k_int = subjname
            k_padded = subjname

        t1_files = list(sub.glob('*_t1.nii*'))
        t2_files = list(sub.glob('*_t2.nii*'))

        any_found = False
        entry = {}
        if t1_files:
            entry['t1'] = str(t1_files[0])
            any_found = True
        if t2_files:
            entry['t2'] = str(t2_files[0])
            any_found = True

        if any_found:
            for key in (subjname, k_int, k_padded):
                prep_map.setdefault(key, {})
                prep_map[key].update(entry)

    return prep_map


def create_combined_csv(ixi_t1_root: str | None,
                        ixi_t2_root: str | None,
                        ixi_prep_root: str | None,
                        xls_path: str,
                        output_csv: str,
                        recursive: bool = False,
                        dry_run: bool = False) -> pd.DataFrame:
    logging.info('Reading metadata: %s', xls_path)
    meta_df = _read_metadata(xls_path)
    id_col, age_col = _detect_cols(meta_df)

    # Normalize metadata ID to string
    meta_df[id_col] = meta_df[id_col].astype(str)

    # Build age lookup by raw id string
    age_lookup = dict(zip(meta_df[id_col].astype(str), meta_df[age_col]))

    # Build file maps
    t1_map = _build_file_maps_from_roots(Path(ixi_t1_root), recursive=recursive) if ixi_t1_root else {}
    t2_map = _build_file_maps_from_roots(Path(ixi_t2_root), recursive=recursive) if ixi_t2_root else {}
    prep_map = _build_prep_map_from_subject_folders(Path(ixi_prep_root)) if ixi_prep_root else {}

    records = []
    for _, row in meta_df.iterrows():
        raw_id = str(row[id_col]).strip()
        padded = raw_id.zfill(3)
        intkey = None
        try:
            intkey = str(int(raw_id))
        except Exception:
            pass

        # helpers to query maps using multiple key variants
        def _get_from_map(m, key_variants):
            for k in key_variants:
                if not k:
                    continue
                v = m.get(k)
                if v:
                    return v
            return ''

        key_variants = [raw_id, padded, intkey]

        # Prefer values present directly in the metadata XLS (if provided)
        xls_t1_orig = ''
        xls_t2_orig = ''
        xls_t1_prep = ''
        xls_t2_prep = ''
        if 'T1_orig' in row.index and not pd.isna(row.get('T1_orig')):
            xls_t1_orig = str(row.get('T1_orig')).strip()
        if 'T2_orig' in row.index and not pd.isna(row.get('T2_orig')):
            xls_t2_orig = str(row.get('T2_orig')).strip()
        if 'T1' in row.index and not pd.isna(row.get('T1')):
            xls_t1_prep = str(row.get('T1')).strip()
        if 'T2' in row.index and not pd.isna(row.get('T2')):
            xls_t2_prep = str(row.get('T2')).strip()

        # Resolve original paths: prefer XLS, else look up from provided roots
        t1_orig = xls_t1_orig if xls_t1_orig else (_get_from_map(t1_map, key_variants) if t1_map else '')
        t2_orig = xls_t2_orig if xls_t2_orig else (_get_from_map(t2_map, key_variants) if t2_map else '')

        # Resolve preprocessed paths: prefer XLS, else look up from prep root
        prep_entry = prep_map.get(raw_id) or prep_map.get(padded) or (prep_map.get(intkey) if intkey else None)
        t1_prep = xls_t1_prep if xls_t1_prep else (prep_entry.get('t1') if prep_entry else '')
        t2_prep = xls_t2_prep if xls_t2_prep else (prep_entry.get('t2') if prep_entry else '')

        age = age_lookup.get(raw_id, '')

        # Enforce presence: ID and AGE always required
        missing_reasons = []
        if not raw_id:
            missing_reasons.append('ID')

        if pd.isna(age) or age in (None, ''):
            missing_reasons.append('AGE')

        have_orig_roots = bool(ixi_t1_root and ixi_t2_root)
        have_prep_root = bool(ixi_prep_root)

        # T1/T2 original roots only > T1/T2 images pair required
        if have_orig_roots:
            if not (t1_orig and t2_orig):
                if not t1_orig:
                    missing_reasons.append('original T1')
                if not t2_orig:
                    missing_reasons.append('original T2')

        # Preprocessed image root only > require complete preprocessed pair
        elif have_prep_root and not have_orig_roots:
            # If the XLS contains T1_orig and T2_orig columns we require that pair to be present
            if 'T1_orig' in row.index and 'T2_orig' in row.index:
                if not (xls_t1_orig and xls_t2_orig):
                    missing_reasons.append('original T1 and T2 paths from CSV')

        # No roots provided > ID and age only, retain paths
        else:
            pass

        if missing_reasons:
            logging.info("Skipping subject %s: missing %s", raw_id or '<no-id>', ", ".join(missing_reasons))
            continue

        records.append({
            'PatientID': raw_id,
            'Age': age,
            'T1_orig': t1_orig,
            'T2_orig': t2_orig,
            'T1': t1_prep,
            'T2': t2_prep,
        })

    out_df = pd.DataFrame(records)
    if dry_run:
        logging.info('[DRY-RUN] Would write CSV with %d subjects to %s', len(out_df), output_csv)
        logging.info('[DRY-RUN] First 5 rows:\n%s', out_df.head().to_string())
    else:
        out_df.to_csv(output_csv, index=False)
        logging.info('Wrote CSV with %d subjects to %s', len(out_df), output_csv)
    return out_df


def _build_arg_parser():
    p = argparse.ArgumentParser(description='Combine IXI metadata and image paths into a single CSV')
    p.add_argument('--ixi_t1_root', default=None, help='path to original IXI T1 files (optional)')
    p.add_argument('--ixi_t2_root', default=None, help='path to original IXI T2 files (optional)')
    p.add_argument('--ixi_prep_root', default=None, help='path to preprocessed IXI folders (optional)')
    p.add_argument('--xls_path', default='IXI.xls', help='path to IXI metadata (Excel or CSV)')
    p.add_argument('--output_csv', default='IXI_full.csv', help='output CSV path')
    p.add_argument('--recursive', action='store_true', help='search image roots recursively')
    p.add_argument('--dry-run', action='store_true', help='preview output without writing CSV')
    p.add_argument('--verbose', action='store_true', help='enable debug logging')
    return p


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')

    create_combined_csv(
        ixi_t1_root=args.ixi_t1_root,
        ixi_t2_root=args.ixi_t2_root,
        ixi_prep_root=args.ixi_prep_root,
        xls_path=args.xls_path,
        output_csv=args.output_csv,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )
