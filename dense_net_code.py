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



# Correct the slice paths to match the actual local directory structure

df['slice_path'] = df['slice_path'].apply(lambda x: os.path.join(data_dir, os.path.basename(x)))



print(f"Total slices: {len(df)}")

print("Tumor Presence Distribution:")

print(df['target'].value_counts())



# Group by volume to do patient-wise split

volumes = df['volume'].unique()

train_vols, temp_vols = train_test_split(volumes, test_size=0.3, random_state=42)

val_vols, test_vols = train_test_split(temp_vols, test_size=0.5, random_state=42)



train_df = df[df['volume'].isin(train_vols)].reset_index(drop=True)

val_df = df[df['volume'].isin(val_vols)].reset_index(drop=True)

test_df = df[df['volume'].isin(test_vols)].reset_index(drop=True)



print(f"Train slices: {len(train_df)} | Val slices: {len(val_df)} | Test slices: {len(test_df)}")

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

                # Based on typical BraTS .h5 prep, keys are usually 'image'

                if 'image' in f.keys():

                    img = np.array(f['image'])

                else:

                    # Fallback if specific key is different

                    key = list(f.keys())[0]

                    img = np.array(f[key])

        except Exception as e:

            # Fallback for corrupted/missing files: create a zero tensor

            img = np.zeros((240, 240, 4), dtype=np.float32)



        # Ensure image is (Channels, Height, Width)

        # BraTS images might be (H, W, C). Let's check and transpose if needed

        if img.shape[-1] <= 4:

            img = np.transpose(img, (2, 0, 1))



        # DenseNet expects 3 channels. If we have 4 (FLAIR, T1w, t1gd, T2w), we can take first 3

        # If 1 channel, we repeat to 3

        if img.shape[0] >= 4:

            img = img[:3, :, :]

        elif img.shape[0] == 1:

            img = np.repeat(img, 3, axis=0)

        elif img.shape[0] == 2:

            img = np.concatenate([img, img[:1, :, :]], axis=0)



        # Normalize to [0, 1] range if not already

        if img.max() > 1.0:

            img = img / 255.0



        img = torch.tensor(img, dtype=torch.float32)



        if self.transform:

            img = self.transform(img)



        target = torch.tensor(target, dtype=torch.float32)



        return img, target



# Define transforms (resizing to match DenseNet requirements)

transform = transforms.Compose([

    transforms.Resize((224, 224), antialias=True),

])



# Create DataLoaders

train_dataset = BraTSDataset(train_df, transform=transform)

val_dataset = BraTSDataset(val_df, transform=transform)

test_dataset = BraTSDataset(test_df, transform=transform)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



print("DataLoaders are ready!")

# Load pre-trained DenseNet121

model = models.densenet121(pretrained=True)



# Replace the classifier for binary classification

num_ftrs = model.classifier.in_features

model.classifier = nn.Sequential(

    nn.Linear(num_ftrs, 1),

    nn.Sigmoid()  # Sigmoid for binary cross entropy

)



model = model.to(device)



criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)



print("Model built and transferred to device.")

epochs = 10



best_val_loss = float('inf')



for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0



    print(f"Epoch {epoch+1}/{epochs}")

    # Train

    for images, labels in tqdm(train_loader, desc="Training"):

        images, labels = images.to(device), labels.to(device)



        optimizer.zero_grad()



        outputs = model(images).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item() * images.size(0)



        # Calculate accuracy

        predicted = (outputs > 0.5).float()

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



    train_loss = running_loss / len(train_loader.dataset)

    train_acc = correct / total



    # Validation

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0



    with torch.no_grad():

        for images, labels in tqdm(val_loader, desc="Validation"):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()

            loss = criterion(outputs, labels)



            val_loss += loss.item() * images.size(0)

            predicted = (outputs > 0.5).float()

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()



    val_loss = val_loss / len(val_loader.dataset)

    val_acc = val_correct / val_total



    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")



    # Save the best model

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), 'best_densenet121_brats.pth')

        print("--> Saved best model!")

    print("-" * 50)

def predict_patient_volume(model, dataframe, volume_id, transform, device):

    """

    Predicts whether a patient has a tumor by analyzing all slices in their volume.

    If a certain threshold of slices predict tumor, we classify the patient as having a tumor.

    """

    model.load_state_dict(torch.load('best_densenet121_brats.pth'))

    model.eval()



    vol_df = dataframe[dataframe['volume'] == volume_id].reset_index(drop=True)

    if len(vol_df) == 0:

        print(f"Volume {volume_id} not found!")

        return



    vol_dataset = BraTSDataset(vol_df, transform=transform)

    vol_loader = DataLoader(vol_dataset, batch_size=16, shuffle=False)



    predictions = []

    actuals = []



    with torch.no_grad():

        for images, labels in tqdm(vol_loader, desc=f"Predicting Volume {volume_id}"):

            images = images.to(device)

            outputs = model(images).squeeze()



            # Ensure output is iterable even if batch size is 1

            if outputs.ndim == 0:

                outputs = outputs.unsqueeze(0)



            predicted = (outputs > 0.5).float()

            predictions.extend(predicted.cpu().numpy())

            actuals.extend(labels.numpy())



    num_tumor_slices = sum(predictions)

    total_slices = len(predictions)

    tumor_percentage = (num_tumor_slices / total_slices) * 100



    true_tumor_slices = sum(actuals)



    print(f"Patient Volume: {volume_id}")

    print(f"Total Slices analyzed: {total_slices}")

    print(f"Predicted Tumor Slices: {num_tumor_slices} ({tumor_percentage:.2f}%)")

    print(f"Actual Tumor Slices: {true_tumor_slices}")



    # If more than 5% of slices have a tumor, classify volume as Positive

    if tumor_percentage > 5.0:

        print(">>> Final Diagnosis: TUMOR DETECTED")

    else:

        print(">>> Final Diagnosis: NO TUMOR")



# Example Usage: Predict for a patient from the test set

sample_test_volume = test_df['volume'].iloc[0]

predict_patient_volume(model, df, sample_test_volume, transform, device)

