import ants
import pandas as pd
from pathlib import Path


class ANTSPyCoregistrationPipeline:
    """
    ANTsPy-based T1-only MRI registration pipeline.
    Uses ANTsPy library instead of command-line tools.
    """

    def __init__(self, csv_path, template_path, output_dir):
        self.csv_path = csv_path
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.df = None
        self._prepare_dirs()

    def _prepare_dirs(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def register_subject(self, row):
        """
        Perform T1 registration workflow for a single subject using ANTsPy.
        """
        subject_id = row['PatientID']
        t1_path = row['T1']

        print(f"\n--- Processing Subject: {subject_id} ---")
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)

        try:
            # Load images
            print("Loading T1 image...")
            t1_img = ants.image_read(str(t1_path))

            print("Loading template...")
            template_img = ants.image_read(str(self.template_path))

            # Step 1: N4 bias correction
            print("Applying N4 bias correction...")
            t1_n4 = ants.n4_bias_field_correction(t1_img)
            n4_output = subject_dir / f"{subject_id}_native_space_t1_n4.nii.gz"
            ants.image_write(t1_n4, str(n4_output))

            # Step 2: Register T1 to template (rigid transformation)
            print("Registering T1 to template...")
            registration = ants.registration(
                fixed=template_img,
                moving=t1_n4,
                type_of_transform='Rigid'
            )

            # Step 3: Save registered image
            print("Saving registered image...")
            final_t1 = subject_dir / f"{subject_id}_SRI_t1.nii.gz"
            ants.image_write(registration['warpedmovout'], str(final_t1))

            # Save transformation matrix
            transform_file = subject_dir / f"{subject_id}_t1_to_template_transform.mat"
            # ANTsPy saves transforms automatically during registration

            print(f"✓ Successfully processed {subject_id}")
            print(f"  Output: {final_t1}")

        except Exception as e:
            print(f"✗ Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()

    def run(self, start_row=0, end_row=None):
        """
        Execute the ANTsPy registration pipeline for a range of subjects.
        """
        print("Loading CSV...")
        self.df = pd.read_csv(self.csv_path)

        if 'PatientID' not in self.df.columns or 'T1' not in self.df.columns:
            raise ValueError(f"Missing required columns in CSV")

        df_slice = self.df.iloc[start_row:end_row]
        print(f"Processing {len(df_slice)} subjects (rows {start_row} to {end_row or 'end'})")

        for idx, row in df_slice.iterrows():
            self.register_subject(row)

        print("\n=== All subjects processed ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANTsPy T1-only MRI registration pipeline")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--template", required=True, help="Path to template image (e.g., MNI)")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--start_row", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--end_row", type=int, help="End row index (exclusive)")

    args = parser.parse_args()

    pipeline = ANTSPyCoregistrationPipeline(
        csv_path=args.csv,
        template_path=args.template,
        output_dir=args.output,
    )
    pipeline.run(start_row=args.start_row, end_row=args.end_row)