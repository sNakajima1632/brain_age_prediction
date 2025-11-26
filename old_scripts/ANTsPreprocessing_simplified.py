import shutil
import subprocess
import pandas as pd
from pathlib import Path

class ANTSCoregistrationPipeline:
    """
    ANTs Multi-contrast MRI co-registration pipeline using ANTs tools.
    
    This class performs the following for each subject:
    - N4 bias correction on native images
    - Co-registration of T1, T2, FLAIR to T1-post
    - Registration of T1-post to a standard template
    - Application of transformations to all contrasts
    - Optional skull stripping using antsBrainExtraction
    - Optional N4 bias correction on skull-stripped images

    Parameters:
    -----------
    template_path : str or Path
        Path to the target anatomical template (e.g., MNI152.nii.gz)
    output_dir : str or Path
        Directory where all outputs and intermediate files will be saved
    brain_template : str or Path, optional
        Path to brain extraction template for skull stripping
    brain_prob_mask : str or Path, optional
        Path to brain probability mask used during skull stripping
    enable_skullstrip : bool, default=True
        If True, skull-stripping will be applied to T1 image after alignment
    enable_n4 : bool, default=True
        If True, apply final N4 bias correction to skull-stripped images
    """
    
    def __init__(self,csv_path, template_path, output_dir,
                 brain_template=None, brain_prob_mask=None,
                 enable_skullstrip=True, enable_n4=True):
        self.csv_path = csv_path
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"  # Temporary folder for intermediate files

        self.brain_template = brain_template
        self.brain_prob_mask = brain_prob_mask
        self.enable_skullstrip = enable_skullstrip
        self.enable_n4 = enable_n4

        self.df = None  # Placeholder for the input CSV
        self.log_file = None  # Log file to track executed commands
        self._prepare_dirs()

    def _prepare_dirs(self):
        """Create output and temp directories if they don't already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def _run_command(self, cmd):
        """Execute a shell command and log it if a log file is set."""
        print(f"Running: {cmd}")
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(cmd + '\n')
        subprocess.run(cmd, shell=True, check=True)

    def n4_bias_correct_image(self, input_image, output_path, normalize=False):
        """
        Apply N4 bias field correction to an image.

        Parameters:
        -----------
        input_image : str
            Path to input image
        output_path : str
            Path where corrected image will be saved
        normalize : bool
            If True, normalize input image intensity before N4
        """
        temp_dir = Path(output_path).parent

        if normalize:
            # Normalize intensity to a range [10, 100] for N4
            norm_img = temp_dir / "norm_input.nii.gz"
            print("---- NORMALIZING ----")
            norm_cmd = f"ImageMath 3 {norm_img} RescaleImage {input_image} 10 100"
            self._run_command(norm_cmd)

        cmd = f"N4BiasFieldCorrection -d 3 -i {input_image} -o {output_path} --verbose"
        self._run_command(cmd)

        if not Path(output_path).exists():
            raise FileNotFoundError(f'N4BiasFieldCorrection failed to create: {output_path}')

    def run_n4_on_images(self, image_dict, subject_id, out_dir, skull_stripped_images=True):
        """
        Apply N4 correction to a dictionary of images.

        Parameters:
        -----------
        image_dict : dict
            Keys are image types (e.g., 't1', 'flair'), values are file paths
        subject_id : str
            ID of the subject (used in output naming)
        out_dir : Path
            Output directory for corrected images
        skull_stripped_images : bool
            Whether images are skull-stripped (affects output file names)
        
        Returns:
        --------
        corrected : dict
            Dictionary of N4-corrected image paths
        """
        corrected = {}
        for name, img_path in image_dict.items():
            suffix = "_brain_n4" if skull_stripped_images else "_n4"
            output_path = out_dir / f"{subject_id}_SRI_{name}{suffix}.nii.gz"
            self.n4_bias_correct_image(str(img_path), str(output_path))
            corrected[name] = str(output_path)
        return corrected



    def _apply_affine(self, moving_img, ref_img, affine, output_path):
        """
        Apply affine transformation to an image using ANTs.

        Parameters:
        -----------
        moving_img : str
            Path to moving image
        ref_img : str
            Path to reference image
        affine : str
            Path to affine transformation matrix
        output_path : str
            Path to save transformed image
        """
        cmd = (
            f"antsApplyTransforms -d 3 -i {moving_img} -r {ref_img} "
            f"-t {affine} -o {output_path} -n Linear"
        )
        self._run_command(cmd)

    def register_subject(self, row):
        """
        Perform full registration workflow for a single subject using a row from the CSV.

        Parameters:
        -----------
        row : pd.Series
            A row from the dataframe with keys [PatientID, T1, T2]
        """
        subject_id = row['PatientID']
        native_paths = {
            't1': row['T1'],
            't2': row['T2'],
        }

        print(f"\n--- Processing Subject: {subject_id} ---")
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        temp_subject_dir = self.temp_dir / subject_id
        temp_subject_dir.mkdir(exist_ok=True)
        self.log_file = subject_dir / f"{subject_id}_ants_pipeline_log.txt"

        try:
            # Step 1: N4 correction on raw images
            n4_paths = {}
            for name, img in native_paths.items():
                out = subject_dir / f"{subject_id}_native_space_{name.lower()}_n4.nii.gz"
                self.n4_bias_correct_image(img, out, normalize=True)
                n4_paths[name] = str(out)

            # Step 2: Align t2 to t1
            step1_affines = {}
            for name in [ 't2']:
                prefix = subject_dir / f"{subject_id}_{name.lower()}_to_t1"
                cmd = (
                    f"antsRegistrationSyN.sh -d 3 -f {n4_paths['t1']} -m {n4_paths[name]} "
                    f"-o {prefix}_ -t r"
                )
                self._run_command(cmd)
                affine = f"{prefix}_0GenericAffine.mat"
                step1_affines[name] = affine
                shutil.copy(affine, subject_dir / f"{subject_id}_{name}_to_t1_affine.mat")

            # Step 3: Apply affines to bring all images to t1 space
            coreg_native = {}
            for name in ['t2']:
                out = subject_dir / f"{subject_id}_{name}_to_t1.nii.gz"
                self._apply_affine(native_paths[name], native_paths['t1'], step1_affines[name], out)
                coreg_native[name] = out

            # Copy t1 as reference image
            t1post_out = subject_dir / f"{subject_id}_t1.nii.gz"
            shutil.copy(native_paths['t1'], t1post_out)
            coreg_native['t1'] = t1post_out

            # Step 4: Register t1 to standard template
            prefix = temp_subject_dir / f"{subject_id}_t1_to_template"
            cmd = (
                f"antsRegistrationSyN.sh -d 3 -f {self.template_path} -m {n4_paths['t1']} "
                f"-o {prefix}_ -t r"
            )
            self._run_command(cmd)
            affine_template = f"{prefix}_0GenericAffine.mat"
            shutil.copy(affine_template, subject_dir / f"{subject_id}_t1_to_template_affine.mat")

            # Step 5: Apply template affine to all t1-aligned images
            final_outputs = {}
            for name, img_path in coreg_native.items():
                out = subject_dir / f"{subject_id}_SRI_{name}.nii.gz"
                self._apply_affine(img_path, self.template_path, affine_template, out)
                final_outputs[name] = out


        finally:
            print(f"Cleaning up temp files for subject: {subject_id}")
            shutil.rmtree(temp_subject_dir)
            self.log_file = None

    def run(self, start_row=0, end_row=None):
        """
        Execute the ANTs co-registration pipeline for a range of subjects defined by row indices.

        Parameters:
        -----------
        start_row : int, default=0
            Starting index (inclusive) of subjects to process in the CSV file.
        end_row : int or None, default=None
            Ending index (exclusive). If None, process until the end of the file.

        Raises:
        -------
        ValueError:
            If required columns are missing from the CSV.
        """
        # Load the input CSV
        self.df = pd.read_csv(self.csv_path)

        # Ensure all required columns are present
        required_cols = ['PatientID', 'T1', 'T2']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col} in CSV")

        # Subset the dataframe using row range
        df_slice = self.df.iloc[start_row:end_row]

        # Process each subject in the selected rows
        for _, row in df_slice.iterrows():
            self.register_subject(row)

        print("All subjects processed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANTs multi-contrast MRI co-registration pipeline")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--template", required=True, help="Path to template image (e.g., MNI)")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--brain_template", type=str, help="Brain extraction template for skull stripping")
    parser.add_argument("--start_row", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--end_row", type=int, help="End row index (exclusive)")

    args = parser.parse_args()

    pipeline = ANTSCoregistrationPipeline(
        csv_path=args.csv,
        template_path=args.template,
        output_dir=args.output,
        brain_template=args.brain_template,
    )
    pipeline.run(start_row=args.start_row, end_row=args.end_row)


