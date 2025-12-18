"""Train a 3D DenseNet121 to predict age from preprocessed T1 NIfTI volumes.

Usage example:
  python IXI_T1_MONAI.py --csv ixi_full.csv --img_col T1 --age_col Age --n_samples 200 --batch_size 4 --epochs 20 --model_out models/IXI_T1_DenseNet121.pt

Notes:
- Expects input NIfTI volumes of shape 240x240x155. If a loaded volume has a different shape,
  it will be center-cropped or zero-padded to the target shape.
- Uses MONAI's DenseNet121 with PyTorch and automatic mixed precision (AMP) for GPU efficiency.
- Metric used: Mean Absolute Error (MAE).
"""

import argparse
import logging
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader

from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split


TARGET_SHAPE = (240, 240, 155)


def _parse_args():
    p = argparse.ArgumentParser(description='Train 3D DenseNet121 on IXI T1 images to predict age')
    p.add_argument('--csv', default='ixi_full.csv', help='CSV with image paths and Age')
    p.add_argument('--img_col', default='T1', help='column name with T1 image paths')
    p.add_argument('--age_col', default='Age', help='column name with Age')
    p.add_argument('--n_samples', type=int, default=None, help='number of samples to use (random sample); default uses all')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_out', default='IXI_T1_DenseNet121.pt', help='output model path (.pt)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def center_crop_or_pad(img: np.ndarray, target: tuple) -> np.ndarray:
    """Center-crop or pad a 3D array to target shape.

    img: (H, W, D)
    target: (H_t, W_t, D_t)
    """
    out = np.zeros((*target,), dtype=img.dtype)
    in_shape = img.shape
    # compute start indices for cropping or padding
    starts = [max((in_shape[i] - target[i]) // 2, 0) for i in range(3)]
    copies = [min(in_shape[i], target[i]) for i in range(3)]

    # compute destination start in out
    dest_starts = [max((target[i] - in_shape[i]) // 2, 0) for i in range(3)]

    src_slices = tuple(slice(starts[i], starts[i] + copies[i]) for i in range(3))
    dst_slices = tuple(slice(dest_starts[i], dest_starts[i] + copies[i]) for i in range(3))
    out[dst_slices] = img[src_slices]
    return out


def load_nifti_numpy(path: str) -> np.ndarray:
    """Load, normalize, and preprocess NIfTI file to target shape.
    
    Returns: (1, H, W, D) for MONAI DenseNet121 (channel-first)
    """
    img = nib.load(path).get_fdata().astype(np.float32)
    # if 4D take first volume
    if img.ndim == 4:
        img = img[..., 0]
    # normalize intensities
    mn = np.min(img)
    mx = np.max(img)
    if mx - mn > 0:
        img = (img - mn) / (mx - mn)
    else:
        img = img * 0.0

    img = center_crop_or_pad(img, TARGET_SHAPE)
    # add channel dim at front for PyTorch (1, H, W, D)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


class NIfTIDataset(TorchDataset):
    """Simple dataset for loading NIfTI files and ages."""
    
    def __init__(self, paths, ages):
        """
        Args:
            paths: list of NIfTI file paths
            ages: list of ages (float)
        """
        self.paths = paths
        self.ages = ages
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        age = self.ages[idx]
        img = load_nifti_numpy(path)
        return torch.from_numpy(img).float(), torch.tensor(age, dtype=torch.float32)


def main():
    args = _parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    set_determinism(seed=args.seed)

    # Enable GPU optimizations for maximum performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Read CSV
    df = pd.read_csv(args.csv) if args.csv.endswith('.csv') else pd.read_excel(args.csv)
    if args.img_col not in df.columns or args.age_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {args.img_col} and {args.age_col}")

    # Filter out missing entries and non-existing files
    df = df.dropna(subset=[args.img_col, args.age_col])
    df[args.img_col] = df[args.img_col].astype(str)
    df['exists'] = df[args.img_col].apply(lambda p: Path(p).exists())
    df = df[df['exists']]
    df = df.reset_index(drop=True)

    if df.empty:
        raise RuntimeError('No valid samples found after filtering missing files/ages')

    # sample limit
    if args.n_samples and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    paths = df[args.img_col].tolist()
    ages = df[args.age_col].astype(float).tolist()
    
    logging.info(f'Loaded {len(paths)} samples')

    # Train/val split
    X_train_p, X_val_p, y_train, y_val = train_test_split(
        paths, ages, test_size=args.val_split, random_state=args.seed
    )
    
    logging.info(f'Train: {len(X_train_p)} | Val: {len(X_val_p)}')

    # Create datasets and dataloaders
    train_ds = NIfTIDataset(X_train_p, y_train)
    val_ds = NIfTIDataset(X_val_p, y_val)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Build model
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
    model = model.to(device)
    
    # Print model summary
    logging.info(f'Model: {model.__class__.__name__}')
    logging.info(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # Use automatic mixed precision (AMP) for GPU speedup and memory efficiency
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None

    best_val_mae = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0

        for imgs, ages in train_loader:
            imgs = imgs.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    preds = model(imgs).squeeze(-1)
                    loss = criterion(preds, ages)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs).squeeze(-1)
                loss = criterion(preds, ages)
                loss.backward()
                optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            n_tr += imgs.size(0)

        if n_tr > 0:
            tr_loss = tr_loss / n_tr
        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs = imgs.to(device)
                ages = ages.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        preds = model(imgs).squeeze(-1)
                else:
                    preds = model(imgs).squeeze(-1)
                val_mae += torch.abs(preds - ages).sum().item()
                n_val += imgs.size(0)

        if n_val > 0:
            val_mae = val_mae / n_val
        
        logging.info(f'Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValMAE {val_mae:.3f}')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()

    # Save best model
    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_model_state is not None:
        torch.save(best_model_state, output_path)
        logging.info(f'Saved best model (ValMAE={best_val_mae:.3f}) to {output_path}')
    else:
        torch.save(model.state_dict(), output_path)
        logging.info(f'Saved model to {output_path}')


if __name__ == '__main__':
    main()
