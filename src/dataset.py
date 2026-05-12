"""
dataset.py
----------
HAM10000 dataset loader with:
- Lesion-ID aware stratified split (train/val/test = 80/10/10)
- WeightedRandomSampler for minority class oversampling
- Weighted cross-entropy class weights (w_i ∝ 1/f_i)
- Standard augmentation for training

Classes (7):
  akiec  Actinic Keratosis   3.3%  HIGH-risk
  bcc    Basal Cell Carc.    5.1%  HIGH-risk
  bkl    Benign Keratosis    11.0% low-risk
  df     Dermatofibroma      1.1%  low-risk
  mel    Melanoma            11.1% HIGH-risk
  nv     Melanocytic Nevi    66.9% low-risk (majority)
  vasc   Vascular Lesion     1.4%  low-risk
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# ── Label mapping ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
HIGH_RISK_CLASSES = {"akiec", "bcc", "mel"}

# HAM10000 channel statistics (precomputed from full dataset)
HAM_MEAN = [0.763, 0.546, 0.570]
HAM_STD  = [0.141, 0.152, 0.170]


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224):
    """Return torchvision transforms for train / val / test."""
    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=HAM_MEAN, std=HAM_STD),
        ])
    else:  # val / test — no augmentation
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=HAM_MEAN, std=HAM_STD),
        ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Parameters
    ----------
    df          : DataFrame with columns [image_id, dx, image_path]
    transform   : torchvision transform pipeline
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Data preparation ───────────────────────────────────────────────────────────
def find_image_path(image_id: str, image_dirs: list[str]) -> str | None:
    """Search for image_id.jpg across multiple image directories."""
    for d in image_dirs:
        path = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def load_metadata(metadata_csv: str, image_dirs: list[str]) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV, resolve image paths, and drop missing images.

    Returns a DataFrame with columns: [image_id, lesion_id, dx, image_path]
    """
    df = pd.read_csv(metadata_csv)
    df["image_path"] = df["image_id"].apply(
        lambda iid: find_image_path(iid, image_dirs)
    )
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[dataset] Warning: {missing} images not found — dropping.")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    return df[["image_id", "lesion_id", "dx", "image_path"]]


def stratified_split(df: pd.DataFrame,
                     train_ratio: float = 0.8,
                     val_ratio:   float = 0.1,
                     seed:        int   = 42):
    """
    Stratified split by LESION_ID (not image_id) to prevent leakage.
    Patients with multiple images are kept entirely in one split.
    """
    # Unique lesions with their class label (take first image's label per lesion)
    lesion_df = df.groupby("lesion_id")["dx"].first().reset_index()

    # Split lesion IDs
    test_ratio = 1.0 - train_ratio - val_ratio
    train_ids, temp_ids = train_test_split(
        lesion_df["lesion_id"],
        test_size=(val_ratio + test_ratio),
        stratify=lesion_df["dx"],
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=lesion_df.set_index("lesion_id").loc[temp_ids, "dx"],
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_ids)]
    val_df   = df[df["lesion_id"].isin(val_ids)]
    test_df  = df[df["lesion_id"].isin(test_ids)]

    print(f"[dataset] Split sizes — train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_class_weights(df: pd.DataFrame,
                          num_classes: int = 7,
                          device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute class weights w_i ∝ 1/f_i, normalised so Σ w_i = num_classes.
    Used for weighted cross-entropy loss.
    """
    counts  = df["dx"].value_counts()
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls, idx in CLASS_TO_IDX.items():
        weights[idx] = 1.0 / counts.get(cls, 1)
    # Normalise so sum = num_classes
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    WeightedRandomSampler: each sample's weight = 1 / class_count.
    Ensures balanced class exposure per batch.
    """
    counts   = df["dx"].value_counts().to_dict()
    sample_w = df["dx"].map(lambda c: 1.0 / counts[c]).values
    sampler  = WeightedRandomSampler(
        weights     = torch.tensor(sample_w, dtype=torch.float64),
        num_samples = len(df),
        replacement = True,
    )
    return sampler


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(metadata_csv:  str,
                    image_dirs:    list[str],
                    batch_size:    int   = 32,
                    image_size:    int   = 224,
                    train_ratio:   float = 0.8,
                    val_ratio:     float = 0.1,
                    num_workers:   int   = 4,
                    seed:          int   = 42,
                    device:        torch.device = torch.device("cpu")):
    """
    Full pipeline: load → split → Dataset → DataLoader.

    Returns
    -------
    loaders : dict with keys "train", "val", "test"
    class_weights : torch.Tensor for weighted cross-entropy
    """
    df = load_metadata(metadata_csv, image_dirs)
    train_df, val_df, test_df = stratified_split(df, train_ratio, val_ratio, seed)

    class_weights = compute_class_weights(train_df, device=device)
    sampler       = make_weighted_sampler(train_df)

    datasets = {
        "train": HAM10000Dataset(train_df, transform=get_transforms("train", image_size)),
        "val":   HAM10000Dataset(val_df,   transform=get_transforms("val",   image_size)),
        "test":  HAM10000Dataset(test_df,  transform=get_transforms("test",  image_size)),
    }

    loaders = {
        "train": DataLoader(datasets["train"],
                            batch_size=batch_size,
                            sampler=sampler,         # WeightedRandomSampler
                            num_workers=num_workers,
                            pin_memory=True),
        "val":   DataLoader(datasets["val"],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True),
        "test":  DataLoader(datasets["test"],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True),
    }

    return loaders, class_weights
