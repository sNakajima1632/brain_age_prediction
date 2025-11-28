"""
Brain Age Training (Windows-Safe, Preprocessed Images)
- CSV columns: img, age  (or: T1 + Age if using T1 only)
- Minimal transforms because images are already preprocessed
- Avoids CacheDataset to prevent OpenMP issues on Windows
"""

import os, argparse, json, random
from pathlib import Path
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism


# ---------------------- ARGUMENTS ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Brain Age Training (Preprocessed + Windows Safe)")
    ap.add_argument("--csv", required=True, help='CSV or Excel file with image paths and age')
    ap.add_argument("--img_col", default="T1", help='column name with T1 image paths')
    ap.add_argument("--age_col", default="Age", help='column name with Age')
    ap.add_argument("--out", default=None, help='output directory or model path (if ends with .pt will save to file)')
    ap.add_argument("--model_out", default=None, help='(optional) explicit model output path (.pt)')
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2, help='validation split ratio')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--n_samples', type=int, default=None, help='limit to N random samples')
    ap.add_argument('--target_shape', nargs=3, type=int, default=[240,240,155], help='target H W D')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


# ---------------------- CSV READER ----------------------
def read_csv_rows(csv_path, img_col, age_col):
    # Keep a robust CSV reader but support pandas for convenience
    import pandas as pd
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV/Excel not found: {csv_path}")

    if p.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    if img_col not in df.columns or age_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {img_col} and {age_col}")

    df = df[[img_col, age_col]].dropna()
    # cast to str/float
    df[img_col] = df[img_col].astype(str).str.strip()
    # filter missing/invalid ages
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    df[age_col] = df[age_col].apply(safe_float)
    df = df.dropna(subset=[age_col])

    rows = []
    for _, r in df.iterrows():
        rows.append({"img": r[img_col], "age": float(r[age_col])})

    if len(rows) == 0:
        raise ValueError("CSV is empty after filtering invalid rows.")

    return rows


# ---------------------- SPLIT ----------------------
def split_train_val(items, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = max(1, int(n * val_ratio))
    return items[n_val:], items[:n_val]


# ---------------------- NIfTI LOADER ----------------------
def center_crop_or_pad_np(img: np.ndarray, target):
    # img: (H,W,D) -> center-crop or pad
    in_shape = img.shape
    out = np.zeros(tuple(target), dtype=img.dtype)
    starts = [max((in_shape[i] - target[i]) // 2, 0) for i in range(3)]
    copies = [min(in_shape[i], target[i]) for i in range(3)]
    dest_starts = [max((target[i] - in_shape[i]) // 2, 0) for i in range(3)]
    src_slices = tuple(slice(starts[i], starts[i] + copies[i]) for i in range(3))
    dst_slices = tuple(slice(dest_starts[i], dest_starts[i] + copies[i]) for i in range(3))
    out[dst_slices] = img[src_slices]
    return out


def _read_nifti(path, target_shape=(240,240,155)):
    import nibabel as nib
    arr = nib.load(path).get_fdata().astype(np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    # normalize
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn > 0:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = arr * 0.0
    arr = center_crop_or_pad_np(arr, tuple(target_shape))
    # return as channel-first (1, D, H, W) expected by MONAI DenseNet121 (it expects channel-first)
    # We'll return (1, H, W, D) then later move axes in collate as needed
    return arr.astype(np.float32, copy=False)


# ---------------------- BATCH COLLATE ----------------------
def collate(batch):
    # batch items may have 'img' as numpy arrays (H,W,D) or (1,H,W,D)
    imgs = []
    ages = []
    for b in batch:
        img = b['img']
        if isinstance(img, np.ndarray):
            # ensure shape (C, D, H, W) for MONAI DenseNet121
            # our _read_nifti returns (H, W, D)
            img = np.expand_dims(img, 0)  # (1, H, W, D)
            # move to (C, D, H, W)
            img = np.moveaxis(img, -1, 1)  # (1, D, H, W)
        imgs.append(torch.from_numpy(img).float())
        ages.append(float(b['age']))

    imgs = torch.stack(imgs, dim=0)
    ages = torch.tensor(ages, dtype=torch.float32)
    return imgs, ages


# ---------------------- MAIN TRAINING ----------------------
def main():
    args = parse_args()
    set_determinism(args.seed)

    # Enable GPU optimizations for maximum performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # prefer explicit model_out then out
    model_out = args.model_out or args.out
    if model_out is None:
        raise ValueError('Please set --out or --model_out to specify output model path or directory')

    outdir = Path(model_out)
    if outdir.suffix == '':
        outdir.mkdir(parents=True, exist_ok=True)
        best_ckpt = outdir / 'best_brain_age.pt'
        bias_path = outdir / 'bias.json'
    else:
        # model_out is a file path
        outdir.parent.mkdir(parents=True, exist_ok=True)
        best_ckpt = outdir if outdir.suffix == '.pt' else Path(str(outdir) + '.pt')
        bias_path = best_ckpt.with_name('bias.json')

    # Read CSV and filter
    all_items = read_csv_rows(args.csv, args.img_col, args.age_col)

    # remove rows with missing files
    filtered = [it for it in all_items if Path(it['img']).exists()]
    if len(filtered) == 0:
        raise RuntimeError('No valid samples found after filtering missing files/ages')

    # sample limit
    if args.n_samples and args.n_samples < len(filtered):
        random.Random(args.seed).shuffle(filtered)
        filtered = filtered[: args.n_samples]

    train_items, val_items = split_train_val(filtered, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] Train: {len(train_items)} | Val: {len(val_items)}")

    target_shape = tuple(args.target_shape)

    # Load NIfTI with normalization and center-crop/pad applied in _read_nifti
    # Use lambda to properly handle dict items from Dataset
    train_ds = Dataset(
        train_items,
        transform=lambda item: {
            'img': _read_nifti(str(item['img']), target_shape),
            'age': item['age']
        }
    )
    val_ds = Dataset(
        val_items,
        transform=lambda item: {
            'img': _read_nifti(str(item['img']), target_shape),
            'age': item['age']
        }
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # Use automatic mixed precision (AMP) for GPU speedup and memory efficiency
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None

    best_val_mae = float('inf')

    all_preds, all_trues = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0

        for imgs, ages in train_loader:
            # imgs may be numpy arrays in collate; ensure device tensors
            imgs = imgs.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(imgs).squeeze(1)
                    loss = criterion(preds, ages)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs).squeeze(1)
                loss = criterion(preds, ages)
                loss.backward()
                optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            n_tr += imgs.size(0)

        if n_tr > 0:
            tr_loss = tr_loss / n_tr
        scheduler.step()

        # validation
        model.eval()
        val_abs = 0.0
        n_val = 0
        epoch_preds, epoch_trues = [], []

        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs = imgs.to(device)
                ages = ages.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(imgs).squeeze(1)
                else:
                    preds = model(imgs).squeeze(1)
                val_abs += torch.abs(preds - ages).sum().item()
                n_val += imgs.size(0)
                epoch_preds.append(preds.cpu().numpy())
                epoch_trues.append(ages.cpu().numpy())

        if n_val == 0:
            val_mae = float('nan')
        else:
            val_mae = val_abs / n_val

        print(f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValMAE {val_mae:.3f}")

        if n_val > 0:
            all_preds.append(np.concatenate(epoch_preds))
            all_trues.append(np.concatenate(epoch_trues))

        if n_val > 0 and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), str(best_ckpt))

    # BIAS CORRECTION if we have preds
    if len(all_preds) > 0 and len(all_trues) > 0:
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        A = np.vstack([all_trues, np.ones_like(all_trues)]).T
        a, b = np.linalg.lstsq(A, all_preds, rcond=None)[0]
        with open(bias_path, 'w') as f:
            json.dump({'a': float(a), 'b': float(b)}, f, indent=2)

    print("\n[INFO] Training Complete!")
    print("[INFO] Best Val MAE:", best_val_mae)
    print("[INFO] Model saved to:", best_ckpt)


if __name__ == "__main__":
    main()
