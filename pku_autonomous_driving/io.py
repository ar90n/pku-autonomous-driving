from pathlib import Path
from typing import Dict, List
from collections import namedtuple

import numpy as np
import pandas as pd
import cv2

BAD_LIST = [
    "ID_1a5a10365",
    "ID_1db0533c7",
    "ID_53c3fe91a",
    "ID_408f58e9f",
    "ID_4445ae041",
    "ID_bb1d991f6",
    "ID_c44983aeb",
    "ID_f30ebe4d4",
]

DataRecord = namedtuple("DataRecord", "image_id, data")


def _str2coords(s, names=["id", "yaw", "pitch", "roll", "x", "y", "z"]):
    """
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    """
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype("float"))))
        if "id" in coords[-1]:
            coords[-1]["id"] = int(coords[-1]["id"])
    return coords


def _parse_df(df: pd.DataFrame) -> List[DataRecord]:
    return [
        DataRecord(row["ImageId"], _str2coords(row["PredictionString"]))
        for _, row in df.iterrows()
    ]


def load_train_data(
    root: Path = Path("../input"), use_bad_list=True, max_num=None
) -> pd.DataFrame:
    path = root / "pku-autonomous-driving" / "train.csv"
    train = pd.read_csv(path)
    if use_bad_list:
        train = train.loc[~train["ImageId"].isin(BAD_LIST)]

    if max_num is not None:
        train = train[:max_num]

    return _parse_df(train)


def load_image(image_id: str, root: Path = Path("../input"), training: bool = True):
    data_type = "train" if training else "test"
    path = root / "pku-autonomous-driving" / f"{data_type}_images" / f"{image_id}.jpg"
    img = cv2.imread(str(path))
    return img


def load_test_data(root: Path = Path("../input")) -> pd.DataFrame:
    path = root / "pku-autonomous-driving" / "sample_submission.csv"
    test = pd.read_csv(path)
    return _parse_df(test)


def load_camera_matrix(root: Path = Path("../input")) -> np.array:
    camera_matrix = np.array(
        [[2304.5479, 0, 1686.2379], [0, 2305.8757, 1354.9849], [0, 0, 1]],
        dtype=np.float32,
    )
    return camera_matrix
