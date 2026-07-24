
import os, cv2, torch, timm, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
def apply_clahe(img_path, img_size=224):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Standard preprocessing for fundus images — enhances
    microaneurysms and hemorrhages that DINO needs to attend to.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # Convert to LAB, apply CLAHE only on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

# Visualize preprocessing effect
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, row in df.sample(4, random_state=42).iterrows():
    path = f"{BASE}/train_images/{row['id_code']}.png"
    raw = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (224,224))
    clahe_img = apply_clahe(path)
    col = list(df.sample(4, random_state=42).index).index(i)
    axes[0][col].imshow(raw)
    axes[0][col].set_title(f"Raw — {class_names[row['diagnosis']]}")
    axes[0][col].axis('off')
    axes[1][col].imshow(clahe_img)
    axes[1][col].set_title(f"CLAHE — Grade {row['diagnosis']}")
    axes[1][col].axis('off')

plt.suptitle('Raw vs CLAHE Preprocessed Fundus Images', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, img_size=224):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = f"{self.img_dir}/{row['id_code']}.png"
        img = apply_clahe(path, self.img_size)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        # Normalize to tensor
        img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std

        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        return img, label


# Augmentations — aggressive for medical imaging
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
])

# Train/val split — stratified to preserve class balance
train_df, val_df = train_test_split(df, test_size=0.15,
                                     stratify=df['diagnosis'],
                                     random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

IMG_DIR = f'{BASE}/train_images'
train_ds = RetinopathyDataset(train_df, IMG_DIR, transform=train_aug)
val_ds   = RetinopathyDataset(val_df,   IMG_DIR, transform=None)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoaders ready ✓")