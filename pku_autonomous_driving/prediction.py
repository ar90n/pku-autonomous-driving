from pathlib import Path
import torch
from torch import nn
from typing import List
from tqdm import tqdm_notebook as tqdm
from .dataset import CarDataset
from .io import DataRecord
from .util import extract_coords, coords2str
import gc

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

use_apex = torch.cuda.is_available() and has_apex

def predict(model, loader, device):
    model.eval()

    result = []
    corrdss = []
    with torch.no_grad():
        for data in loader:
            img = data["img"].to(device)
            predicts = model(img).data.cpu().numpy()

            for out in predicts:
                coords = extract_coords(data, out)
                corrdss.append(coords)
                result.append(coords2str(coords))
    return result, corrdss


def clean_up():
    torch.cuda.empty_cache()
    gc.collect()


def setup(model: nn.Module, optimizer, device, path: Path = None, opt_level='O1'):
    checkpoint = {}
    if path is not None:
        checkpoint = torch.load(str(path))

    if use_apex:
        model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])

    if not use_apex:
        model.to(device)
    return model, optimizer
