import torch
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
from .dataset import CarDataset
from .io import DataRecord
from .util import extract_coords, coords2str
import gc

def predict(model, input, org_shape):
    predictions = []
    with torch.no_grad():
        output = model(input)
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out, org_shape)
        s = coords2str(coords)
        predictions.append(s)
    return predictions

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()


def create_data_loader(test: List[DataRecord]):
    test_dataset = CarDataset(test, training=False)
    return DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
