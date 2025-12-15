import shutil
import pandas as pd
from pathlib import Path
import os
import re
import ants
import tempfile
import SimpleITK as sitk

class ANTSCoregistrationPipeline:
    """
    ANTs Multi-contrast MRI co-registration pipeline using antspyx.
    
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
    t1_path : str or Path, optional
        Path to T1-weighted images directory
    t2_path : str or Path, optional
        Path to T2-weighted images directory
    brain_template : str or Path, optional
        Path to brain extraction template for skull stripping
    brain_prob_mask : str or Path, optional
        Path to brain probability mask used during skull stripping
    enable_skullstrip : bool, default=True
        If True, skull-stripping will be applied to T1 image after alignment
    enable_n4 : bool, default=True
        If True, apply final N4 bias correction to skull-stripped images
    """

    def __init__(self, csv_path, template_path, output_dir, t1_path, t2_path,
                 brain_template=None, brain_prob_mask=None,
                 enable_skullstrip=True, enable_n4=True):

        self.csv_path = csv_path
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        self.t1_path = Path(t1_path) if t1_path else None
        self.t2_path = Path(t2_path) if t2_path else None

        self.brain_template = brain_template
        self.brain_prob_mask = brain_prob_mask
        self.enable_skullstrip = enable_skullstrip
        self.enable_n4 = enable_n4

        self.df = None      # Placeholder for the input CSV
        self._prepare_dirs()

    def _prepare_dirs(self):
        """Create output and temp directories if they don't already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    #  N4 BIAS CORRECTION  (ANTsPy)
    def n4_bias_correct_image(self, input_image, output_path, normalize=False):
        """
        Apply N4 bias field correction using ants.n4_bias_field_correction.

        Parameters:
        -----------
        input_image : str
            Path to input image
        output_path : str
            Path where corrected image will be saved
        normalize : bool
            If True, normalize input image intensity before N4
        """
        print(f"[N4]  {input_image}")

        if normalize:
            # Normalize intensities to range 10–100 for N4
            print("---- NORMALIZING ----")
            sitk_img = sitk.ReadImage(str(input_image))
            rescaled = sitk.RescaleIntensity(sitk_img, 0.0, 100.0)
            # Save temporary normalized image
            tmp = Path(output_path).with_suffix('.rescaled.nii.gz')
            sitk.WriteImage(rescaled, str(tmp))
            img = ants.image_read(str(tmp))
        else:
            img = ants.image_read(str(input_image))

        corrected = ants.n4_bias_field_correction(img)
        ants.image_write(corrected, str(output_path))

        if not Path(output_path).exists():
            raise FileNotFoundError(f"N4 ANTsPy failed: {output_path}")

    #  APPLY AFFINE TRANSFORM (ANTsPy)
    def _apply_affine(self, moving_img, ref_img, affine, output_path):
        """
        Apply affine transformation using ants.apply_transforms.

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

        moving = ants.image_read(str(moving_img))
        reference = ants.image_read(str(ref_img))

        transformed = ants.apply_transforms(
            fixed=reference,
            moving=moving,
            transformlist=[affine],  # affine dict/ANTsPy object
            interpolator="linear",
        )
        ants.image_write(transformed, str(output_path))

    # ----------------------------------------------------------
    #  MAIN SUBJECT REGISTRATION WORKFLOW
    # ----------------------------------------------------------
    def register_subject(self, row):
        subject_id = str(row[self.id_col])
        print(f"\n=== PROCESSING SUBJECT {subject_id} ===")

        # Use a temporary directory for intermediates; only final SRI outputs
        # will be written to self.output_dir/subject_id.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tempdir = tempfile.TemporaryDirectory()
        subj_temp = Path(tempdir.name)

        # -------------------------
        # LOAD INPUT IMAGES
        # -------------------------
        native_paths = {
            "t1": row.get("T1_orig"),
            "t2": row.get("T2_orig")
        }

        # -------------------------
        # STEP 1 — N4 correction
        # -------------------------
        n4_paths = {}
        for name, img in native_paths.items():
            if not img or pd.isna(img):
                continue
            out = subj_temp / f"{subject_id}_native_{name}_n4.nii.gz"
            self.n4_bias_correct_image(img, out, normalize=True)
            n4_paths[name] = str(out)

        # ----------------------------------------------------------
        # STEP 2 — REGISTER T2 → T1  (ANTS REGISTRATION)
        # ----------------------------------------------------------
        print(f"Registering T2 → T1")

        fixed_t1 = ants.image_read(n4_paths["t1"])
        moving_t2 = ants.image_read(n4_paths["t2"])

        reg_t2_to_t1 = ants.registration(
            fixed=fixed_t1,
            moving=moving_t2,
            type_of_transform="Affine"     # matches SYN -t r (rigid/affine)
        )

        # Save affine transform for consistency
        affine_t2_to_t1 = reg_t2_to_t1["fwdtransforms"][0]

        # ----------------------------------------------------------
        # STEP 3 — APPLY AFFINE TO RAW T2
        # ----------------------------------------------------------
        print("Applying affine to bring T2 into T1 space...")

        out_t2_t1 = subj_temp / f"{subject_id}_t2_to_t1.nii.gz"
        self._apply_affine(
            native_paths["t2"],
            native_paths["t1"],
            affine_t2_to_t1,
            out_t2_t1
        )

        coreg_native = {
            "t1": Path(n4_paths.get("t1") or native_paths.get("t1")),
            "t2": out_t2_t1
        }

        # ----------------------------------------------------------
        # STEP 4 — REGISTER T1 to TEMPLATE
        # ----------------------------------------------------------
        print("Registering T1 → TEMPLATE")

        template = ants.image_read(str(self.template_path))
        moving_t1 = ants.image_read(n4_paths["t1"])

        reg_t1_to_template = ants.registration(
            fixed=template,
            moving=moving_t1,
            type_of_transform="Affine"
        )

        affine_t1_to_template = reg_t1_to_template["fwdtransforms"][0]

        # ----------------------------------------------------------
        # STEP 5 — APPLY TEMPLATE AFFINE TO ALL COREGISTERED IMAGES
        # ----------------------------------------------------------
        print("Applying T1→TEMPLATE transform to all modalities...")

        # create a subject-specific output directory and save outputs inside it
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        final_paths = {}
        for name, path in coreg_native.items():
            # Save T1 as <subject_id>.nii.gz and other contrasts with a suffix
            if name == 't1':
                out = subject_dir / f"{subject_id}_SRI_{name}_preprocessed.nii.gz"
            else:
                out = subject_dir / f"{subject_id}_SRI_{name}_preprocessed.nii.gz"
            self._apply_affine(
                path,
                self.template_path,
                affine_t1_to_template,
                out
            )
            final_paths[name] = str(out)

        # Clean up intermediates
        try:
            tempdir.cleanup()
        except Exception:
            pass

        # Record final output paths back into the dataframe (if loaded)
        try:
            idx = row.name
            if hasattr(self, 'df') and idx in self.df.index:
                self.df.at[idx, 'SRI_T1'] = final_paths.get('t1', '')
                self.df.at[idx, 'SRI_T2'] = final_paths.get('t2', '')
        except Exception:
            pass

        print(f"✓ DONE: {subject_id}")

    # ----------------------------------------------------------
    #  RUNNER
    # ----------------------------------------------------------
    def run(self, start_row=0, end_row=None):
        """
        Execute the ANTsPy registration pipeline for a range of subjects.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
        except:
            self.df = pd.read_excel(self.csv_path)

        # detect id column (any column containing 'id')
        id_col = next((c for c in self.df.columns if 'id' in c.lower()), None)
        if id_col is None:
            raise ValueError('No column containing "id" found in CSV')
        self.id_col = id_col

        # Ensure required modality columns exist in dataframe or will be added
        # If t1/t2 directories are provided, expect raw T1/T2 paths in the CSV
        if self.t1_path is None and 'T1_orig' not in self.df.columns:
            raise ValueError('Missing T1 information: either provide t1_path or T1 column in CSV')
        if self.t2_path is None and 'T2_orig' not in self.df.columns:
            raise ValueError('Missing T2 information: either provide t2_path or T2 column in CSV')

        df_slice = self.df.iloc[start_row:end_row]

        for _, row in df_slice.iterrows():
            self.register_subject(row)

        print("=== All subjects processed. ===")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ANTsPy Multi-contrast MRI Co-registration Pipeline"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to input CSV with subject information")
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to anatomical template (e.g., MNI)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs")
    parser.add_argument("--t1_path", type=str, default=None,
                        help="Directory containing T1-weighted images")
    parser.add_argument("--t2_path", type=str, default=None,
                        help="Directory containing T2-weighted images")
    parser.add_argument("--start_row", type=int, default=0,
                        help="Start row index (inclusive)")
    parser.add_argument("--end_row", type=int, default=None,
                        help="End row index (exclusive)")
    
    args = parser.parse_args()

    pipeline = ANTSCoregistrationPipeline(
        csv_path=args.csv_path,
        template_path=args.template_path,
        output_dir=args.output_dir,
        t1_path=args.t1_path,
        t2_path=args.t2_path
    )
    pipeline.run(start_row=args.start_row, end_row=args.end_row)