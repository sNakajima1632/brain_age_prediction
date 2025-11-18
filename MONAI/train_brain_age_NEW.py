"""
Brain Age (GPU/CPU) with MONAI DenseNet121 3D — stable pipeline
- CSV 列: img, age
- 读取 NIfTI: nibabel -> numpy.float32
- 预处理: Add channel -> CropForeground -> ScaleIntensity -> Resize
  （先不做 Orientation/Spacing，跑通后再加）
- 3D DenseNet121 回归（MAE），保存最佳验证模型
- 简单偏差校正（基于验证集）
"""

import os, csv, argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.data import CacheDataset
from monai.transforms import (
    Compose, Lambdad, EnsureChannelFirstd,
    CropForegroundd, ScaleIntensityd, Resized, EnsureTyped
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism


def parse_args():
    ap = argparse.ArgumentParser("Brain Age Training (MONAI DenseNet121 3D)")
    ap.add_argument("--csv", required=True, help="训练 CSV 路径（列: img, age）")
    ap.add_argument("--img_col", default="img", help="图像列名，默认 img")
    ap.add_argument("--age_col", default="age", help="年龄列名，默认 age")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--roi", type=int, nargs=3, default=[160,192,160], help="输入尺寸 D H W")
    return ap.parse_args()


def read_csv_rows(csv_path, img_col, age_col):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        assert img_col in r.fieldnames, f"CSV 找不到列: {img_col}, 可用列: {r.fieldnames}"
        assert age_col in r.fieldnames, f"CSV 找不到列: {age_col}, 可用列: {r.fieldnames}"
        for row in r:
            p = row[img_col].strip()
            a = float(row[age_col])
            rows.append({"img": p, "age": a})
    assert len(rows) > 0, "CSV 为空"
    return rows


def split_train_val(items, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items); n_val = max(1, int(n * val_ratio))
    return items[n_val:], items[:n_val]


# ---------- I/O：用 nibabel 读取，返回纯 numpy.float32 ----------
def _read_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32, copy=False)
    return arr


def make_transforms(roi):
    roi = tuple(int(x) for x in roi)
    common = [
        Lambdad(keys=["img"], func=_read_nifti),               # 读成 numpy.float32
        EnsureChannelFirstd(keys=["img"]),                     # (1, D, H, W)
        CropForegroundd(keys=["img"], source_key="img"),
        ScaleIntensityd(keys=["img"]),                         # z-score / [0,1] 线性缩放（按 MONAI 默认）
        Resized(keys=["img"], spatial_size=roi),
        EnsureTyped(keys=["img"], dtype=torch.float32, track_meta=False),  # 转 torch，关闭 meta
    ]
    return Compose(common), Compose(common)


def collate(batch):
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    ages = torch.tensor([b["age"] for b in batch], dtype=torch.float32)
    return imgs, ages


def main():
    args = parse_args()
    set_determinism(args.seed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    all_items = read_csv_rows(args.csv, args.img_col, args.age_col)
    train_items, val_items = split_train_val(all_items, args.val_ratio, args.seed)
    print(f"[INFO] Train: {len(train_items)} | Val: {len(val_items)}")

    train_tfms, val_tfms = make_transforms(args.roi)
    train_ds = CacheDataset(train_items, transform=train_tfms, cache_rate=1.0, num_workers=args.num_workers)
    val_ds   = CacheDataset(val_items,   transform=val_tfms,   cache_rate=1.0, num_workers=args.num_workers)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            print("[INFO] Using GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, args.epochs))

    best_val_mae = float("inf")
    best_ckpt = outdir / "densenet3d_t1_best.pt"

    # ---- Train Loop ----
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss, n_tr = 0.0, 0
        for imgs, ages in train_loader:
            imgs, ages = imgs.to(device, non_blocking=True), ages.to(device, non_blocking=True)
            preds = model(imgs).squeeze(1)
            loss = criterion(preds, ages)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            n_tr += imgs.size(0)
        scheduler.step()
        tr_loss /= max(1, n_tr)

        # Validate
        model.eval()
        val_abs, n_val = 0.0, 0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device, non_blocking=True), ages.to(device, non_blocking=True)
                preds = model(imgs).squeeze(1)
                val_abs += torch.abs(preds - ages).sum().item()
                n_val += imgs.size(0)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(ages.cpu().numpy())
        val_mae = val_abs / max(1, n_val)
        print(f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValMAE {val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_ckpt)

    # ---- Bias Correction (on validation set) ----
    all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
    A = np.vstack([all_trues, np.ones_like(all_trues)]).T
    a, b = np.linalg.lstsq(A, all_preds, rcond=None)[0]  # pred ≈ a*true + b
    with open(outdir / "bias_correction.json", "w") as f:
        json.dump({"a": float(a), "b": float(b), "note": "use pred'=(pred-b)/a at inference"}, f, indent=2)

    print(f"[INFO] Best Val MAE: {best_val_mae:.3f}")
    print(f"[INFO] Saved model: {best_ckpt}")
    print(f"[INFO] Saved bias params: {outdir/'bias_correction.json'}")


if __name__ == "__main__":
    main()