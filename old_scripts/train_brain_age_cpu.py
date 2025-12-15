"""
Brain Age Training (Windows-Safe, Preprocessed Images)
- CSV columns: img, age  (or: T1 + Age if using T1 only)
- Minimal transforms because images are already preprocessed
- Avoids CacheDataset to prevent OpenMP issues on Windows
"""

import os, csv, argparse, json, random
from pathlib import Path
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose, Lambdad, EnsureChannelFirstd,
    EnsureTyped
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism


# ---------------------- ARGUMENTS ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Brain Age Training (Preprocessed + Windows Safe)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_col", default="T1")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------- CSV READER ----------------------
def read_csv_rows(csv_path, img_col, age_col):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)

        for row in r:
            img_path = row[img_col].strip()
            age_value = str(row[age_col]).strip()

            # --- Skip missing or invalid age ---
            if age_value == "" or age_value.lower() == "nan":
                print(f"[WARN] Skipping row: missing age for image {img_path}")
                continue

            try:
                age_float = float(age_value)
            except ValueError:
                print(f"[WARN] Skipping row: invalid age '{age_value}' for image {img_path}")
                continue

            rows.append({
                "img": img_path,
                "age": age_float
            })

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
def _read_nifti(path):
    import nibabel as nib
    arr = nib.load(path).get_fdata()
    return arr.astype(np.float32, copy=False)


# ---------------------- TRANSFORMS (MINIMAL) ----------------------
def make_transforms():
    common = [
        Lambdad(keys=["img"], func=_read_nifti),
        EnsureChannelFirstd(keys=["img"]),             # (1, D, H, W)
        EnsureTyped(keys=["img"], dtype=torch.float32)
    ]
    return Compose(common), Compose(common)


# ---------------------- BATCH COLLATE ----------------------
def collate(batch):
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    ages = torch.tensor([b["age"] for b in batch], dtype=torch.float32)
    return imgs, ages


# ---------------------- MAIN TRAINING ----------------------
def main():
    args = parse_args()
    set_determinism(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # DATA
    all_items = read_csv_rows(args.csv, args.img_col, args.age_col)
    train_items, val_items = split_train_val(all_items, args.val_ratio, args.seed)
    print(f"[INFO] Train: {len(train_items)} | Val: {len(val_items)}")

    train_tfms, val_tfms = make_transforms()

    # Windows-safe dataset
    train_ds = Dataset(train_items, transform=train_tfms)
    val_ds   = Dataset(val_items, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=0,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch,
                              shuffle=False, num_workers=0,
                              collate_fn=collate)

    # MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=1
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    best_ckpt = outdir / "best_brain_age.pt"

    # TRAINING LOOP
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0
        n = 0

        for imgs, ages in train_loader:
            imgs, ages = imgs.to(device), ages.to(device)
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        tr_loss /= n
        scheduler.step()

        # VALIDATION
        model.eval()
        val_abs = 0
        n = 0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device), ages.to(device)
                preds = model(imgs).squeeze(1)
                val_abs += torch.abs(preds - ages).sum().item()
                n += imgs.size(0)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(ages.cpu().numpy())

        val_mae = val_abs / n
        print(f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValMAE {val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_ckpt)

    # BIAS CORRECTION
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    A = np.vstack([all_trues, np.ones_like(all_trues)]).T
    a, b = np.linalg.lstsq(A, all_preds, rcond=None)[0]

    with open(outdir / "bias.json", "w") as f:
        json.dump({"a": float(a), "b": float(b)}, f, indent=2)

    print("\n[INFO] Training Complete!")
    print("[INFO] Best Val MAE:", best_val_mae)
    print("[INFO] Model saved to:", best_ckpt)


if __name__ == "__main__":
    main()
