import os

import h5py

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

from PIL import Image

import warnings

warnings.filterwarnings('ignore')



# Set GPU Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

# Load Metadata

metadata_path = 'BraTS20 Training Metadata.csv'

data_dir = 'BraTS2020_training_data/content/data'



df = pd.read_csv(metadata_path)



# Fix paths

df['slice_path'] = df['slice_path'].apply(lambda x: os.path.join(data_dir, os.path.basename(x)))



print(f"Total slices: {len(df)}")

print(df['target'].value_counts())



# -------------------------------

# PATIENT-WISE SAMPLING (CRITICAL)

# -------------------------------

all_volumes = df['volume'].unique()



# Shuffle for randomness

np.random.seed(42)

np.random.shuffle(all_volumes)



# Select 70 for training, 30 for testing

train_vols = all_volumes[:70]

test_vols = all_volumes[70:100]   # next 30 patients



# From 70 â†’ split train/val

train_vols, val_vols = train_test_split(train_vols, test_size=0.2, random_state=42)



# Create dataframes

train_df = df[df['volume'].isin(train_vols)].reset_index(drop=True)

val_df = df[df['volume'].isin(val_vols)].reset_index(drop=True)

test_df = df[df['volume'].isin(test_vols)].reset_index(drop=True)



print(f"Train patients: {len(train_vols)}")

print(f"Val patients: {len(val_vols)}")

print(f"Test patients: {len(test_vols)}")



print(f"Train slices: {len(train_df)}")

print(f"Val slices: {len(val_df)}")

print(f"Test slices: {len(test_df)}") 
class BraTSDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe

        self.transform = transform



    def __len__(self):

        return len(self.dataframe)



    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        file_path = row['slice_path']

        target = row['target']



        try:

            with h5py.File(file_path, 'r') as f:

                key = 'image' if 'image' in f.keys() else list(f.keys())[0]

                img = np.array(f[key])

        except:

            img = np.zeros((240, 240, 4), dtype=np.float32)



        # Convert shape â†’ (C, H, W)

        if img.shape[-1] <= 4:

            img = np.transpose(img, (2, 0, 1))



        # Fix channels

        if img.shape[0] >= 4:

            img = img[:3]

        elif img.shape[0] == 1:

            img = np.repeat(img, 3, axis=0)

        elif img.shape[0] == 2:

            img = np.concatenate([img, img[:1]], axis=0)



        # Normalize

        img = img.astype(np.float32)

        if img.max() > 1:

            img /= 255.0



        img = torch.tensor(img)



        if self.transform:

            img = self.transform(img)



        target = torch.tensor(target, dtype=torch.float32)



        return img, target
transform_train = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

])



transform_val = transforms.Compose([

    transforms.Resize((224, 224)),

])
train_dataset = BraTSDataset(train_df, transform=transform_train)

val_dataset = BraTSDataset(val_df, transform=transform_val)

test_dataset = BraTSDataset(test_df, transform=transform_val)



train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,

                          num_workers=2, pin_memory=True)



val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,

                        num_workers=2, pin_memory=True)



test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,

                         num_workers=2, pin_memory=True)



print("DataLoaders ready (RAM optimized)") 
model = models.densenet121(weights='IMAGENET1K_V1')



num_ftrs = model.classifier.in_features

model.classifier = nn.Sequential(

    nn.Linear(num_ftrs, 1),

    nn.Sigmoid()

)



model = model.to(device)



criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(),lr=5e-5, weight_decay=1e-4) #3e-4



# LR Scheduler (IMPORTANT)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5) 
model2 = models.densenet121(weights='IMAGENET1K_V1')



num_ftrs = model2.classifier.in_features

model.classifier = nn.Linear(num_ftrs, 1)



model = model.to(device)



criterion = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer, T_max=30, eta_min=1e-6

)
epochs = 15

best_val_loss = float('inf')



for epoch in range(epochs):

    model.train()

    train_loss, correct, total = 0, 0, 0



    for images, labels in tqdm(train_loader):

        images, labels = images.to(device), labels.to(device)



        optimizer.zero_grad()

        outputs = model(images).squeeze()



        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        train_loss += loss.item() * images.size(0)



        preds = (outputs > 0.5).float()

        correct += (preds == labels).sum().item()

        total += labels.size(0)



    train_loss /= len(train_loader.dataset)

    train_acc = correct / total



    # VALIDATION

    model.eval()

    val_loss, correct, total = 0, 0, 0



    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()



            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)



            preds = (outputs > 0.5).float()

            correct += (preds == labels).sum().item()

            total += labels.size(0)



    val_loss /= len(val_loader.dataset)

    val_acc = correct / total



    scheduler.step(val_loss)



    print(f"\nEpoch {epoch+1}")

    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")



    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), 'best_model.pth')

        print("âœ… Model Saved")



    torch.cuda.empty_cache()
