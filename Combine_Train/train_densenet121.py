import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from pathlib import Path
from tqdm import tqdm

set_determinism(42)

# -----------------------
# 1. Dataset
# -----------------------
class BrainAgeDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        age = torch.tensor([float(row["Age"])], dtype=torch.float32)

        img_path = row["Path"]  

        img = nib.load(img_path).get_fdata()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img, age


# -----------------------
# 2. Load Data
# -----------------------
ROOT = "/home/blue/Blue_Project/Combine_Train"   # ★ 已修改

train_csv = f"{ROOT}/train_split.csv"
val_csv   = f"{ROOT}/val_split.csv"
test_csv  = f"{ROOT}/test_split.csv"

train_ds = BrainAgeDataset(train_csv)
val_ds   = BrainAgeDataset(val_csv)
test_ds  = BrainAgeDataset(test_csv)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4)


# -----------------------
# 3. Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=1  # regression
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# AMP scaler
scaler = torch.cuda.amp.GradScaler()


# -----------------------
# 4. Train & Validate
# -----------------------
EPOCHS = 50
best_val = float("inf")
save_path = f"{ROOT}/best_model.pt"

def evaluate(loader):
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for x, age in loader:
            x = x.to(device)
            age = age.to(device)
            pred = model(x)
            loss = criterion(pred, age)
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for x, age in loop:
        x = x.to(device)
        age = age.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criterion(pred, age)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * x.size(0)
        loop.set_postfix({"train_loss": loss.item()})

    train_loss /= len(train_ds)
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: Train MAE={train_loss:.3f}, Val MAE={val_loss:.3f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"🔥 New best model saved (Val MAE={val_loss:.3f})")


# -----------------------
# 5. Test
# -----------------------
print("\nLoading best model for testing...")
model.load_state_dict(torch.load(save_path))
test_loss = evaluate(test_loader)
print(f"🎯 Test MAE = {test_loss:.3f} years")
