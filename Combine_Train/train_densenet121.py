import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from monai.transforms import (
    Compose, 
    Resize, 
    ScaleIntensity,
    EnsureChannelFirst,
    ToTensor
)
from pathlib import Path
from tqdm import tqdm

set_determinism(42)

# -----------------------
# 1. Dataset with Transforms
# -----------------------
class BrainAgeDataset(Dataset):
    def __init__(self, csv_path, root_dir="../", target_size=(160, 192, 160)):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target_size = target_size
        
        # ★ 定义transforms
        self.transforms = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),  # 归一化到[0,1]
            Resize(spatial_size=target_size, mode='trilinear'),  # resize到固定大小
        ])
        
        print(f"Loading {csv_path}: {len(self.df)} samples")
        print(f"Root directory: {self.root_dir}")
        print(f"Target size: {self.target_size}")
        assert not self.df['Age'].isna().any(), "Age contains NaN values!"
        assert not self.df['Path'].isna().any(), "Path contains NaN values!"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        age = float(row["Age"])

        img_path = row["Path"]
        
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 加载图像
        nii_img = nib.load(img_path)
        img = nii_img.get_fdata(dtype=np.float32)
        
        # 转换为tensor并添加channel维度
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W, D]
        
        # ★ 应用transforms（归一化+resize）
        img = self.transforms(img)
        
        age_tensor = torch.tensor([age], dtype=torch.float32)

        return img, age_tensor


# -----------------------
# ★ 新增：早停类
# -----------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience: 多少个epoch没有改善后停止
            min_delta: 认为是改善的最小变化量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"Initial best loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered!")
        else:
            if self.verbose:
                print(f"Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0


# -----------------------
# 2. Load Data
# -----------------------
ROOT = "../Combine_Train"

train_csv = f"{ROOT}/train_split.csv"
val_csv   = f"{ROOT}/val_split.csv"
test_csv  = f"{ROOT}/test_split.csv"

# 检查CSV文件是否存在
for csv_file in [train_csv, val_csv, test_csv]:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

train_ds = BrainAgeDataset(train_csv)
val_ds   = BrainAgeDataset(val_csv)
test_ds  = BrainAgeDataset(test_csv)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)


# -----------------------
# 3. Model
# -----------------------
if torch.cuda.is_available():
    device = torch.device("cuda")  # 自动使用CUDA_VISIBLE_DEVICES指定的GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("WARNING: CUDA is not available! Using CPU instead.")
    

model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=1  # regression
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5, verbose=True  # ★ 增加patience
)

# AMP scaler
scaler = torch.cuda.amp.GradScaler()


# -----------------------
# 4. Train & Validate
# -----------------------
EPOCHS = 500  
best_val = float("inf")
save_path = f"{ROOT}/best_model.pt"
last_checkpoint_path = f"{ROOT}/last_model.pt"  # ★ 保存最后一个模型

# 创建checkpoint目录
os.makedirs(ROOT, exist_ok=True)

# ★ 初始化早停
early_stopping = EarlyStopping(patience=15, min_delta=0.001, verbose=True)

# ★ 用于记录训练历史
train_losses = []
val_losses = []

def evaluate(loader):
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for x, age in loader:
            x = x.to(device)
            age = age.to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, age)
            
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


print(f"\n{'='*60}")
print(f"Starting Training: {EPOCHS} epochs with early stopping")
print(f"Early stopping patience: {early_stopping.patience}")
print(f"Learning rate scheduler patience: 5")
print(f"{'='*60}\n")

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
    
    # ★ 记录历史
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Train MAE={train_loss:.3f}, Val MAE={val_loss:.3f}, LR={current_lr:.2e}")

    # 保存最佳模型
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, save_path)
        print(f"🔥 New best model saved (Val MAE={val_loss:.3f})")
    
    # ★ 每个epoch保存最后一个模型（用于恢复训练）
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val': best_val,
    }, last_checkpoint_path)
    
    # ★ 检查早停
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\n{'='*60}")
        print(f"Early stopping at epoch {epoch}")
        print(f"Best validation MAE: {best_val:.3f}")
        print(f"{'='*60}\n")
        break
    
    print()  # 空行分隔


# -----------------------
# ★ 保存训练历史
# -----------------------
history_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses
})
history_df.to_csv(f"{ROOT}/training_history.csv", index=False)
print(f"Training history saved to {ROOT}/training_history.csv")


# -----------------------
# 5. Test
# -----------------------
print("\nLoading best model for testing...")
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model from epoch {checkpoint['epoch']}, Val MAE={checkpoint['val_loss']:.3f}")

test_loss = evaluate(test_loader)
print(f"🎯 Test MAE = {test_loss:.3f} years")

# ★ 保存测试结果
with open(f"{ROOT}/test_result.txt", 'w') as f:
    f.write(f"Best model epoch: {checkpoint['epoch']}\n")
    f.write(f"Best validation MAE: {checkpoint['val_loss']:.3f}\n")
    f.write(f"Test MAE: {test_loss:.3f}\n")
print(f"\nTest results saved to {ROOT}/test_result.txt")