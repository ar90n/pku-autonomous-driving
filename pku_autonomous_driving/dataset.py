from pathlib import Path
from typing import Union, Iterable
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
import torch
from torch.utils.data import Dataset

from .io import load_image, DataRecord


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

    def _pack_to_dict(self, img, data):
        return {"img": img, "affine_mat": np.eye(3, 3), "data": data}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        record = self.dataset[idx]

        # Read image
        kwargs = {"training": self.training}
        if self.root is not None:
            kwargs["root"] = self.root
        img0 = load_image(record.image_id, **kwargs)

        input = self._pack_to_dict(img0, record.data)
        if self.transform:
            input = self.transform(input)

        return input


def create_data_loader(dataset: Union[Iterable[Dataset], Dataset], **kwargs):
    if isinstance(dataset, Iterable):
        dataset = ConcatDataset(dataset)
    return DataLoader(dataset=dataset, **kwargs)
