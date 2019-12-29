from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import load_image
from .improc import preprocess_image, get_mask_and_regr


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe: pd.DataFrame, root: Path = None, training=True, transform=None):
        self.df = dataframe
        self.root = root
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        
        # Read image
        kwargs = {}
        if self.root is not None:
            kwargs['root'] = self.root
        img0 = load_image(idx, **kwargs)
        img = preprocess_image(img0)
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        if self.training:
            mask, regr = get_mask_and_regr(img0, labels)
            regr = np.rollaxis(regr, 2, 0)
        else:
            mask, regr = 0, 0
        
        return [img, mask, regr]
