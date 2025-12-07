#!/usr/bin/env python3
"""
Combine IXI and CoRR datasets into a single CSV file.

IXI dataset (ixi_full.csv):
  - Columns: PatientID, Age, T1_orig, T2_orig, T1, T2
  - T1/T2 are preprocessed paths (need to add prefix)
  - Keep both T1 and T2

CoRR dataset (CoRR_Age_Training_Table.csv):
  - Columns: PatientID, Age, PT_Path
  - PT_Path becomes T1 (no T2 images)
  - Drop rows with missing/non-numeric Age

Output (IXI_CoRR_combined.csv):
  - Columns: PatientID, Age, T1, T2
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='Combine IXI and CoRR datasets into a single CSV'
    )
    p.add_argument(
        '--ixi-csv',
        default='../ixi_full.csv',
        help='Path to IXI CSV file (default: ixi_full.csv)'
    )
    p.add_argument(
        '--corr-csv',
        default='../CoRR_Age_Training_Table.csv',
        help='Path to CoRR CSV file (default: CoRR_Age_Training_Table.csv)'
    )
    p.add_argument(
        '--ixi-prefix',
        default='/home/blue/Blue_Project/',
        help='Prefix to add to IXI T1/T2 paths (default: /home/blue/Blue_Project/)'
    )
    p.add_argument(
        '--output',
        default='../IXI_CoRR_combined.csv',
        help='Output CSV file (default: IXI_CoRR_combined.csv)'
    )
    p.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    return p.parse_args()


def process_ixi(csv_path, prefix, verbose=False):
    """
    Process IXI dataset.
    
    Returns DataFrame with columns: PatientID, Age, T1, T2
    """
    print(f"Loading IXI dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"  Original shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['PatientID', 'Age', 'T1', 'T2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"IXI CSV missing columns: {missing_cols}")
    
    # Select and rename columns
    df_processed = df[['PatientID', 'Age', 'T1', 'T2']].copy()
    
    # Add prefix to T1 and T2 paths, convert backslashes to forward slashes
    df_processed['T1'] = (prefix + df_processed['T1'].astype(str)).str.replace('\\', '/')
    df_processed['T2'] = (prefix + df_processed['T2'].astype(str)).str.replace('\\', '/')
    
    # Ensure Age is numeric
    df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce')
    
    # Drop rows with invalid Age
    initial_count = len(df_processed)
    df_processed = df_processed.dropna(subset=['Age'])
    dropped = initial_count - len(df_processed)
    
    if verbose:
        print(f"  Processed shape: {df_processed.shape}")
        if dropped > 0:
            print(f"  Dropped {dropped} rows with missing/invalid Age")
    
    print(f"  ✓ Loaded {len(df_processed)} IXI samples")
    return df_processed


def process_corr(csv_path, verbose=False):
    """
    Process CoRR dataset.
    
    Returns DataFrame with columns: PatientID, Age, T1, T2 (T2 empty)
    """
    print(f"Loading CoRR dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"  Original shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['PatientID', 'Age', 'PT_Path']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CoRR CSV missing columns: {missing_cols}")
    
    # Ensure Age is numeric and drop invalid rows
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    initial_count = len(df)
    df = df.dropna(subset=['Age'])
    dropped = initial_count - len(df)
    
    if verbose:
        print(f"  Dropped {dropped} rows with missing/invalid Age")
    
    # Create output dataframe with T1 = PT_Path and T2 empty
    # Replace .pt extension with _preprocessed.nii.gz and normalize slashes
    t1_paths = df['PT_Path'].astype(str).str.replace(r'\.pt$', '_preprocessed.nii.gz', regex=True)
    t1_paths = t1_paths.str.replace('\\', '/')

    df_processed = pd.DataFrame({
        'PatientID': df['PatientID'],
        'Age': df['Age'],
        'T1': t1_paths,
        'T2': ''  # CoRR has no T2 images
    })
    
    if verbose:
        print(f"  Processed shape: {df_processed.shape}")
    
    print(f"  ✓ Loaded {len(df_processed)} CoRR samples")
    return df_processed


def combine_datasets(ixi_df, corr_df, output_path, verbose=False):
    """
    Combine IXI and CoRR dataframes and save to CSV.
    """
    print(f"\nCombining datasets...")
    
    # Concatenate dataframes
    combined = pd.concat([ixi_df, corr_df], ignore_index=True)
    
    if verbose:
        print(f"  Combined shape: {combined.shape}")
    
    # Save to CSV
    combined.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Combined Dataset Summary")
    print(f"{'='*60}")
    print(f"Total samples:     {len(combined)}")
    print(f"  IXI samples:     {len(ixi_df)}")
    print(f"  CoRR samples:    {len(corr_df)}")
    print(f"Age range:         {combined['Age'].min():.1f} - {combined['Age'].max():.1f}")
    print(f"Age mean:          {combined['Age'].mean():.1f}")
    print(f"Columns:           {list(combined.columns)}")
    print(f"{'='*60}\n")
    
    return combined


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Process IXI dataset
        ixi_df = process_ixi(args.ixi_csv, args.ixi_prefix, args.verbose)
        
        # Process CoRR dataset
        corr_df = process_corr(args.corr_csv, args.verbose)
        
        # Combine and save
        combined = combine_datasets(ixi_df, corr_df, args.output, args.verbose)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
