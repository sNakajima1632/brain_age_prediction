import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from monai.networks.nets import resnet
from monai.utils import set_determinism
from tqdm import tqdm

set_determinism(42)


# ============================
# Dataset
# ============================
class BrainAgeDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # 保证 Age 是数字（已经提前过滤过，但再保险）
        self.df = self.df[pd.to_numeric(self.df["Age"], errors="coerce").notnull()]
        self.df["Age"] = self.df["Age"].astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        age = torch.tensor([row["Age"]], dtype=torch.float32)

        img = torch.load(row["PT_Path"])
        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img, age


# ============================
# Data Loaders
# ============================
ROOT = "/home/blue/Blue_Project/CoRR_Train"

train_ds = BrainAgeDataset(f"{ROOT}/train_split.csv")
val_ds   = BrainAgeDataset(f"{ROOT}/val_split.csv")
test_ds  = BrainAgeDataset(f"{ROOT}/test_split.csv")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)


# ============================
# Model: 3D ResNet18
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet.resnet18(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=1,   # 回归输出 1 个值
).to(device)

criterion = nn.L1Loss()  # MAE
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

# AMP（加速）
scaler = torch.cuda.amp.GradScaler()


# ============================
# Evaluation
# ============================
def evaluate(loader):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for x, age in loader:
            x = x.to(device)
            age = age.to(device)

            pred = model(x)
            loss = criterion(pred, age)

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    return total_loss / count


# ============================
# Training Loop
# ============================
EPOCHS = 70
best_val = float("inf")
save_path = f"{ROOT}/best_resnet18.pt"

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
        print(f"🔥 New best model saved! (Val MAE={val_loss:.3f})")


# ============================
# Final Testing
# ============================
print("\nLoading best model for testing...")
model.load_state_dict(torch.load(save_path))
test_loss = evaluate(test_loader)
print(f"🎯 Test MAE = {test_loss:.3f} years")
