import sys
sys.path.insert(0, r'D:\Python_packages')
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import CLASS_NAMES, set_seed

IMAGE_SIZE = 224

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        self.img_path_map = {}
        for img_dir in img_dirs:
            for fname in os.listdir(img_dir):
                img_id = os.path.splitext(fname)[0]
                self.img_path_map[img_id] = os.path.join(img_dir, fname)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label = CLASS_NAMES[row['dx']]
        img_path = self.img_path_map[img_id]
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

def prepare_dataloaders(data_dir, batch_size=32):
    set_seed(42)
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    df['age'].fillna(df['age'].median(), inplace=True)
    img_dirs = [
        os.path.join(data_dir, 'HAM10000_images_part_1'),
        os.path.join(data_dir, 'HAM10000_images_part_2')
    ]
    unique_lesions = df['lesion_id'].unique()
    train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.30, random_state=42)
    val_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.50, random_state=42)
    train_df = df[df['lesion_id'].isin(train_lesions)]
    val_df   = df[df['lesion_id'].isin(val_lesions)]
    test_df  = df[df['lesion_id'].isin(test_lesions)]
    print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
    class_counts = train_df['dx'].value_counts()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = train_df['dx'].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataset = HAM10000Dataset(train_df, img_dirs, transform=train_transform)
    val_dataset   = HAM10000Dataset(val_df,   img_dirs, transform=val_transform)
    test_dataset  = HAM10000Dataset(test_df,  img_dirs, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, class_weights
