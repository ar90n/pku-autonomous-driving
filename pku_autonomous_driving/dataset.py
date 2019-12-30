from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .io import load_image, DataRecord
from .improc import preprocess_image, get_mask_and_regr


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(
        self,
        dataset: List[DataRecord],
        root: Path = None,
        training=True,
        transform=None,
    ):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        record = self.dataset[idx]

        # Read image
        kwargs = {}
        if self.root is not None:
            kwargs["root"] = self.root
        img0 = load_image(record.image_id, **kwargs)
        img = preprocess_image(img0)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        if self.training:
            mask, regr = get_mask_and_regr(img0, record.data)
            regr = np.rollaxis(regr, 2, 0)
        else:
            mask, regr = 0, 0

        return [img, mask, regr]


def create_train_dataset(
    train: List[DataRecord], test_size: float, random_state: int = 64
):
    train_records, dev_records = train_test_split(
        train, test_size=0.08, random_state=random_state
    )
    train_dataset = CarDataset(train_records)
    dev_dataset = CarDataset(dev_records)
    return train_dataset, dev_dataset