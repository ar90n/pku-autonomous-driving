from pathlib import Path

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


def load_train_data(root: Path = Path("../input"), use_bad_list=True) -> pd.DataFrame:
    path = root / "pku-autonomous-driving" / "train.csv"
    train = pd.read_csv(path)
    if use_bad_list:
        train = train.loc[~train["ImageId"].isin(BAD_LIST)]

    return train


def load_image(image_id: str, root: Path = Path("../input"), training: bool = True):
    data_type = "train" if training else "test"
    path = root / "pku-autonomous-driving" / f"{data_type}_images" / f"{image_id}.jpg"
    img = cv2.imread(str(path))
    return img


def load_test_data(root: Path = Path("../input")) -> pd.DataFrame:
    path = root / "pku-autonomous-driving" / "sample_submission.csv"
    return pd.read_csv(path)


def load_camera_matrix(root: Path = Path("../input")) -> np.array:
    camera_matrix = np.array(
        [[2304.5479, 0, 1686.2379], [0, 2305.8757, 1354.9849], [0, 0, 1]],
        dtype=np.float32,
    )
    return camera_matrix
