import os
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING AND SPLITTING
# ==========================================
metadata_path = 'BraTS20 Training Metadata.csv'
data_dir = 'BraTS2020_training_data/content/data'

print("Loading metadata...")
df = pd.read_csv(metadata_path)

df['slice_path'] = df['slice_path'].apply(
    lambda x: os.path.join(data_dir, os.path.basename(x))
)

all_volumes = df['volume'].unique()
np.random.seed(42)
np.random.shuffle(all_volumes)

# Using a subset of 100 volumes
train_vols = all_volumes[:70]
test_vols = all_volumes[70:100]

train_vols, val_vols = train_test_split(train_vols, test_size=0.2, random_state=42)

train_df = df[df['volume'].isin(train_vols)].reset_index(drop=True)
val_df = df[df['volume'].isin(val_vols)].reset_index(drop=True)

print(f"Train slices: {len(train_df)} | Val slices: {len(val_df)}")


# ==========================================
# 2. DATASET CLASS WITH FIXES
# ==========================================
class BraTSDatasetSeg(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with h5py.File(row['slice_path'], 'r') as f:
            img = np.array(f['image'])
            mask = np.array(f['mask'])

        # -------- IMAGE FIX --------
        # Take first 3 channels (e.g. FLAIR, T1, T1gd)
        img = np.transpose(img, (2, 0, 1))[:3]
        img = img.astype(np.float32)
        
        # Robust Normalization (Z-score on non-zero pixels per channel)
        for c in range(img.shape[0]):
            mask_c = img[c] > 0
            if mask_c.any():
                mean = img[c][mask_c].mean()
                std = img[c][mask_c].std()
                if std > 0:
                    img[c][mask_c] = (img[c][mask_c] - mean) / std

        # -------- MASK FIX (CRITICAL) --------
        # Combine all tumor sub-region channels into a single whole-tumor mask
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=2)

        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 🔥 RESIZE (BIG SPEED BOOST)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(128, 128)).squeeze(0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)   

        return img, mask

train_seg_dataset = BraTSDatasetSeg(train_df)
val_seg_dataset = BraTSDatasetSeg(val_df)

train_seg_loader = DataLoader(
    train_seg_dataset,
    batch_size=29,   # 🔥 increase batch
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
val_seg_loader = DataLoader(val_seg_dataset, batch_size=8)


# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        smooth = 1e-5
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)


# ==========================================
# 4. UNET ARCHITECTURE
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = block(256, 512)

        self.dropout = nn.Dropout(0.3)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.dropout(self.bottleneck(self.pool(e3)))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ==========================================
# 5. TRAINING ROUTINE
# ==========================================
def train_model(epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    unet = UNet().to(device)

    criterion = CombinedLoss()   # 🔥 NEW LOSS
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    # Using updated GradScaler syntax for PyTorch 2.x
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        unet.train()
        train_loss = 0

        # Optional: Add tqdm for progress tracking
        # pbar = tqdm(train_seg_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks in train_seg_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            if masks.dim() == 5:
                masks = masks.squeeze(4)

            # Updated autocast syntax
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                preds = unet(imgs)

                if preds.shape != masks.shape:
                    masks = torch.nn.functional.interpolate(
                        masks, size=preds.shape[2:], mode='nearest'
                    )

                loss = criterion(preds, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {train_loss:.4f}")

    torch.save(unet.state_dict(), "unet_best.pth")
    print("✅ HIGH-QUALITY U-Net Trained and saved to unet_best.pth")


# ==========================================
# 6. VISUALIZATION AND EVALUATION ROUTINE
# ==========================================
def visualize_unet(volume_id, slice_index=50, model_path="unet_best.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    unet = UNet().to(device)
    try:
        unet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"❌ Failed to load model weights from {model_path}: {e}")
        return
        
    unet.eval()

    vol_df = df[df['volume'] == volume_id].reset_index(drop=True)

    if len(vol_df) == 0:
        print("❌ Volume not found")
        return

    if slice_index >= len(vol_df):
        print(f"❌ Slice index {slice_index} out of range (max: {len(vol_df)-1})")
        return

    row = vol_df.iloc[slice_index]

    # LOAD FROM H5
    with h5py.File(row['slice_path'], 'r') as f:
        img = np.array(f['image'])
        mask = np.array(f['mask'])

    # -------- IMAGE FIX --------
    img = np.transpose(img, (2, 0, 1))[:3]
    img = img.astype(np.float32)
    
    # Robust Normalization (Z-score on non-zero pixels per channel)
    for c in range(img.shape[0]):
        mask_c = img[c] > 0
        if mask_c.any():
            mean = img[c][mask_c].mean()
            std = img[c][mask_c].std()
            if std > 0:
                img[c][mask_c] = (img[c][mask_c] - mean) / std

    # -------- MASK FIX --------
    if len(mask.shape) == 3:
        mask = np.max(mask, axis=2)

    mask = (mask > 0).astype(np.float32)

    # -------- RESIZE (same as training) --------
    img_tensor = torch.tensor(img).unsqueeze(0)
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(128, 128))
    mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(128, 128), mode='nearest')

    img_tensor = img_tensor.to(device)

    # -------- PREDICTION --------
    with torch.no_grad():
        pred = unet(img_tensor)
        pred = torch.sigmoid(pred)   # 🔥 IMPORTANT (since logits)

    pred = pred.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()

    # -------- VISUALIZATION --------
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # Apply global min-max scaling for visualization and clip
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    # Apply threshold to output crisp binary mask
    plt.imshow(pred > 0.5, cmap='gray')
    plt.axis('off')

    plt.show()

    # -------- TUMOR AREA --------
    tumor_pixels = np.sum(pred > 0.5)
    print(f"Tumor Pixel Count: {tumor_pixels}")
    print("Pred min:", pred.min(), "max:", pred.max())

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # To run training, uncomment the following line:
    # train_model(epochs=15)
    
    # To run visualization, keep this section active:
    test_volume = 152   # 👈 choose any volume ID
    visualize_unet(test_volume, slice_index=49)
