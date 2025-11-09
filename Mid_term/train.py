import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from shutil import copy2, move
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



## Prepare data for training

# paths
BASE = "/content/oil_spill_dataset"
IMG_SRC = os.path.join(BASE, "images", "images", "train")
MASK_SRC = os.path.join(BASE, "mask", "masks", "train")

OUT = "/content/oil_spill_dataset_prepped"
os.makedirs(OUT, exist_ok=True)

for split in ["train","val","test"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks", split), exist_ok=True)

# list all image files (assuming masks have same filenames)
all_files = sorted([f for f in os.listdir(IMG_SRC) if f.lower().endswith(('.png','.jpg','.jpeg'))])
print("Total train images:", len(all_files))

# split - use 20% of the original train as validation; keep provided val as test later
train_files, val_temp = train_test_split(all_files, test_size=0.20, random_state=42)

# copy splits
for f in train_files:
    copy2(os.path.join(IMG_SRC, f), os.path.join(OUT, "images", "train", f))
    copy2(os.path.join(MASK_SRC, f), os.path.join(OUT, "masks", "train", f))

for f in val_temp:
    copy2(os.path.join(IMG_SRC, f), os.path.join(OUT, "images", "val", f))
    copy2(os.path.join(MASK_SRC, f), os.path.join(OUT, "masks", "val", f))

# copy the provided validation set as test (images/images/val and mask/masks/val)
PROVIDED_VAL_IMG = os.path.join(BASE, "images", "images", "val")
PROVIDED_VAL_MASK = os.path.join(BASE, "mask", "masks", "val")
provided_val_files = sorted([f for f in os.listdir(PROVIDED_VAL_IMG) if f.lower().endswith(('.png','.jpg'))])
for f in provided_val_files:
    copy2(os.path.join(PROVIDED_VAL_IMG, f), os.path.join(OUT, "images", "test", f))
    copy2(os.path.join(PROVIDED_VAL_MASK, f), os.path.join(OUT, "masks", "test", f))

print("Prepared folders under:", OUT)



## creating segdataset class for image data

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files, size=128, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = files
        self.size = size
        self.augment = augment
        self.tf = T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),
            # images are grayscale: ToTensor -> shape (1, H, W)
            T.Normalize([0.5], [0.5])  # single channel norm
        ])
        self.tf_mask = T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),  # mask -> float [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, f)).convert('L')  # 'L' for grayscale
        mask = Image.open(os.path.join(self.mask_dir, f)).convert('L')
        img = self.tf(img)
        mask = self.tf_mask(mask)
        # ensure binary 0/1
        mask = (mask > 0.5).float()
        return img, mask


train_files = sorted(os.listdir("/content/oil_spill_dataset_prepped/images/train"))
val_files = sorted(os.listdir("/content/oil_spill_dataset_prepped/images/val"))

train_ds = SegDataset("/content/oil_spill_dataset_prepped/images/train",
                      "/content/oil_spill_dataset_prepped/masks/train",
                      train_files, size=128)
val_ds = SegDataset("/content/oil_spill_dataset_prepped/images/val",
                    "/content/oil_spill_dataset_prepped/masks/val",
                    val_files, size=128)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)


## Define Tiny Unet model architecture
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

# 2 encoder and decoder unet architecture
class TinyUNet2Layer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_c=16):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_c)
        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_c * 2, base_c * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c * 2, base_c)

        # Final output layer
        self.final = nn.Conv2d(base_c, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        return self.final(d1)
    
def iou_score(pred, target, eps=1e-6):
    # pred, target are binary tensors (B,1,H,W)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=[1,2,3])
    union = ((pred + target) > 0).float().sum(dim=[1,2,3])
    return ((inter + eps) / (union + eps)).mean().item()

from torch.utils.data import ConcatDataset, DataLoader

# existing datasets
train_ds = SegDataset("/content/oil_spill_dataset_prepped/images/train",
                      "/content/oil_spill_dataset_prepped/masks/train",
                      sorted(os.listdir("/content/oil_spill_dataset_prepped/images/train")),
                      size=128)

val_ds = SegDataset("/content/oil_spill_dataset_prepped/images/val",
                    "/content/oil_spill_dataset_prepped/masks/val",
                    sorted(os.listdir("/content/oil_spill_dataset_prepped/images/val")),
                    size=128)

# combine them for final training
trainval_ds = ConcatDataset([train_ds, val_ds])

# new dataloader for training
trainval_loader = DataLoader(trainval_ds, batch_size=16, shuffle=True, num_workers=2)


model = TinyUNet(in_ch=1, out_ch=1, base_c=16).to(device)
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    for imgs, masks in trainval_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
torch.save(model, "oil_spill_unet_full.pth")