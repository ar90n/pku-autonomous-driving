import math

import numpy as np
import cv2

from .const import IMG_WIDTH, IMG_HEIGHT, MODEL_SCALE
from .util import str2coords, get_img_coords
from .geometry import rotate, proj_world_to_screen


def _regr_preprocess(regr_dict):
    for name in ["x", "y", "z"]:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict["roll"] = rotate(regr_dict["roll"], np.pi)
    regr_dict["pitch_sin"] = math.sin(regr_dict["pitch"])
    regr_dict["pitch_cos"] = math.cos(regr_dict["pitch"])
    regr_dict.pop("pitch")
    regr_dict.pop("id")
    return regr_dict


def preprocess_image(img):
    img = img[img.shape[0] // 2 :]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, : img.shape[1] // 4]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return (img / 255).astype("float32")


def get_mask_and_regr(img, data):
    mask = np.zeros(
        [IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype="float32"
    )
    regr = np.zeros(
        [IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype="float32"
    )
    for regr_dict in data:
        world_coords = np.array(
            [regr_dict["x"], regr_dict["y"], regr_dict["z"]]
        ).reshape(-1, 3)
        xs, ys = proj_world_to_screen(world_coords)
        x, y = ys[0], xs[0]
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype("int")
        y = (y + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * 1.5) / MODEL_SCALE
        y = np.round(y).astype("int")
        if (
            x >= 0
            and x < IMG_HEIGHT // MODEL_SCALE
            and y >= 0
            and y < IMG_WIDTH // MODEL_SCALE
        ):
            mask[x, y] = 1
            regr_dict2 = _regr_preprocess({**regr_dict})
            regr[x, y] = [regr_dict2[n] for n in sorted(regr_dict2)]
    return mask, regr
