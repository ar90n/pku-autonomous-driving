import torch
from typing import List
from tqdm import tqdm_notebook as tqdm
from .dataset import CarDataset
from .io import DataRecord
from .util import extract_coords, coords2str
import gc


def predict(model, loader, device):
    model.eval()

    predictions = []
    with torch.no_grad():
        for data in loader:
            img = data["img"].to(device)
            predicts = model(img).data.cpu().numpy()

            for out in predicts:
                coords = extract_coords(data, out)
                s = coords2str(coords)
                predictions.append(s)
    return predictions


def clean_up():
    torch.cuda.empty_cache()
    gc.collect()
