from IXI_csv_paths import create_combined_csv
from IXI_antspyPreprocessing import ANTSCoregistrationPipeline

from Combine_Train.combine_datasets import process_ixi, process_corr, combine_datasets
from Combine_Train.split_combined import split_combined_dataset
from Combine_Train.train_densenet121 import train_densenet121_t1

from T1_T2.split_ixi import split_ixi_dataset
from T1_T2.train import train_densenet121_t1t2

def main () -> None:
    """
    End-to-end, automated pipeline for building and training MONAI DenseNet121
    model for brain age prediction using combined IXI and CoRR datasets.

    Steps:
    1. Preprocess IXI dataset and create metadata CSV.
    2. Preprocess CoRR dataset and create metadata CSV.
    3. Train MONAI DenseNet121 model for T1w images from IXI and CoRR datasets.
    4. Train MONAI DenseNet121 model for T1w+T2w images from IXI dataset.
    """
    # ----------------------------------------------------------
    #  1. IXI dataset: Preprocess and create metadata CSV
    # ----------------------------------------------------------
    
    # Metadata CSV creation.
    # Output: "IXI_full.csv"
    # Extracted data: PatientID, Age, T1 original path, T2 original path
    create_combined_csv(
        ixi_t1_root="IXI-T1",
        ixi_t2_root="IXI-T2",
        xls_path="IXI.xls",
        output_csv="IXI_full.csv"
    )

    # Preprocessing using ANTs registration pipeline.
    # Output: Preprocessed T1 and T2 images in "IXI_prep" directory.
    pipeline = ANTSCoregistrationPipeline(
        csv_path="IXI_full.csv",
        template_path="T1_FeTS.nii.gz",
        output_dir="IXI_prep"
    )
    pipeline.run()

    # Metadata CSV update with preprocessed paths.
    # Output: "IXI_preprocessed.csv"
    # Add preprocessed T1 and T2 image paths to CSV.
    create_combined_csv(
        xls_path="IXI_full.csv",
        ixi_prep_root="IXI_prep",
        output_csv="IXI_full.csv"
    )
    
    # ----------------------------------------------------------
    #  2. CoRR dataset: Preprocess and create metadata CSV
    # ----------------------------------------------------------
    
    """
    Add code here for CoRR dataset preprocessing and CSV creation.
    """   

    # ----------------------------------------------------------
    #  3. T1w Model Training: IXI and CoRR combined dataset
    # ----------------------------------------------------------

    # Combine IXI and CoRR datasets into a single CSV.
    # Output: "IXI_CoRR_combined.csv"
    # parameters
    ixi_csv = "IXI_full.csv"
    corr_csv = "CoRR_Age_Training_Table.csv"
    ixi_prefix = "../"
    output_csv = "IXI_CoRR_combined.csv"

    # run process
    try:
        ixi_df = process_ixi(ixi_csv, ixi_prefix, verbose=True)
        corr_df = process_corr(corr_csv, verbose=True)
        combined = combine_datasets(ixi_df, corr_df, output_csv, verbose=True)
    except Exception as e:
        print("Combine failed:", e)
        return
    
    # Split combined dataset into train/val/test sets.
    # Outputs: "train_split.csv", "val_split.csv", "test_split.csv"
    split_combined_dataset()

    # Train MONAI DenseNet121 model for T1w images.
    # Output: Trained model weights saved to disk.
    train_densenet121_t1()
    
    # ----------------------------------------------------------
    #  4. T1w+T2w Model Training: IXI dataset only
    # ----------------------------------------------------------

    # Split IXI dataset into train/val/test sets for T1w+T2w model.
    # Outputs: "train_split.csv", "val_split.csv", "test_split.csv"
    split_ixi_dataset()

    # Train MONAI DenseNet121 model for T1w+T2w images.
    # Output: Trained model weights saved to disk.
    train_densenet121_t1t2()

if __name__ == "__main__":
    main()