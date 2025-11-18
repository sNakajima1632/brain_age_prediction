import shutil
import pandas as pd
from pathlib import Path
import os
import re
import ants

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

        img = ants.image_read(str(input_image))

        if normalize:
            # Normalize intensities to range 10–100 for N4
            print("---- NORMALIZING ----")
            img = ants.rescale_intensity(img, newrange=(10, 100))

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
        subject_id = str(row["IXI_ID"])
        print(f"\n=== PROCESSING SUBJECT {subject_id} ===")

        subj_dir = self.output_dir / subject_id
        subj_dir.mkdir(exist_ok=True)

        # -------------------------
        # LOAD INPUT IMAGES
        # -------------------------
        native_paths = {
            "t1": row["T1"],
            "t2": row["T2"]
        }

        # -------------------------
        # STEP 1 — N4 correction
        # -------------------------
        n4_paths = {}
        for name, img in native_paths.items():
            out = subj_dir / f"{subject_id}_native_{name}_n4.nii.gz"
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

        out_t2_t1 = subj_dir / f"{subject_id}_t2_to_t1.nii.gz"
        self._apply_affine(
            native_paths["t2"],
            native_paths["t1"],
            affine_t2_to_t1,
            out_t2_t1
        )

        coreg_native = {
            "t1": Path(native_paths["t1"]),
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

        for name, path in coreg_native.items():
            out = subj_dir / f"{subject_id}_SRI_{name}.nii.gz"
            self._apply_affine(
                path,
                self.template_path,
                affine_t1_to_template,
                out
            )

        print(f"✓ DONE: {subject_id}")

    # ----------------------------------------------------------
    #  CSV UPDATE FUNCTION (UNCHANGED)
    # ----------------------------------------------------------
    def update_csv(self, dataframe, col_name=None):
        # UNCHANGED — your original implementation
        # (you said to keep this as-is)
        ...
        # paste your same code here
        ...

    # ----------------------------------------------------------
    #  RUNNER
    # ----------------------------------------------------------
    def run(self, start_row=0, end_row=None):
        """Runs pipeline; unchanged except for removing CLI checks."""
        base, ext = os.path.splitext(self.csv_path)
        updated = Path(f"{base}_updated{ext}")
        if updated.exists():
            self.csv_path = updated

        try:
            self.df = pd.read_excel(self.csv_path)
        except:
            self.df = pd.read_csv(self.csv_path)

        required_cols = ["IXI_ID"]

        if self.t1_path is not None:
            self.df = self.update_csv(self.df, "T1")
        else:
            required_cols.append("T1")

        if self.t2_path is not None:
            self.df = self.update_csv(self.df, "T2")
        else:
            required_cols.append("T2")

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

        df_slice = self.df.iloc[start_row:end_row]

        for _, row in df_slice.iterrows():
            self.register_subject(row)

        print("=== All subjects processed. ===")
