import torch
from torch import nn
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from .const import DISTANCE_THRESH_CLEAR, IMG_WIDTH, IMG_HEIGHT, MODEL_SCALE
from .geometry import proj_world_to_screen, rotate
from .io import load_camera_matrix


def _regr_back(regr_dict):
    regr_dict["z"] = regr_dict["z"] * 100
    regr_dict["roll"] = rotate(regr_dict["roll"], -np.pi)

    pitch_sin = regr_dict["pitch_sin"] / np.sqrt(
        regr_dict["pitch_sin"] ** 2 + regr_dict["pitch_cos"] ** 2
    )
    pitch_cos = regr_dict["pitch_cos"] / np.sqrt(
        regr_dict["pitch_sin"] ** 2 + regr_dict["pitch_cos"] ** 2
    )
    regr_dict["pitch"] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def str2coords(s, names=["id", "yaw", "pitch", "roll", "x", "y", "z"]):
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


def coords2str(coords, names=["yaw", "pitch", "roll", "x", "y", "z", "confidence"]):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return " ".join(s)


def aggregate_coords(data):
    xs = [c["x"] for c in data]
    ys = [c["y"] for c in data]
    zs = [c["z"] for c in data]
    return np.array(list(zip(xs, ys, zs)))


def get_img_coords(data):
    coords = aggregate_coords(data)
    proj_coords = proj_world_to_screen(coords)
    xs = proj_coords[:, 0]
    ys = proj_coords[:, 1]
    return xs, ys


def optimize_xy(r, c, x0, y0, z0, affine_mat, inv_camera_mat):
    proj_coords = np.array([MODEL_SCALE * r, MODEL_SCALE * c, 1])
    #est_pos = ((z0 * affine_mat @ proj_coords)[[1, 0, 2]]) @ inv_camera_mat.T
    est_pos = affine_mat @ proj_coords
    est_pos = a0 * est_pos
    est_pos = est_pos[[1, 0, 2]]
    est_pos = est_pos @ inv_camera_mat.T
    x_new = x0 + est_pos[0]
    y_new = y0 + est_pos[1]
    return x_new, y_new, z0


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1["x"], c1["y"], c1["z"]])
        for c2 in coords:
            xyz2 = np.array([c2["x"], c2["y"], c2["z"]])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1["confidence"] < c2["confidence"]:
                    c1["confidence"] = -1
    return [c for c in coords if c["confidence"] > 0]


def extract_coords(data, prediction=None):
    if prediction is None:
        logits = data["mask"]
        regr_output = data["regr"]
    else:
        logits = prediction[0]
        regr_output = prediction[1:]

    points = np.argwhere(logits > 0)
    col_names = sorted(["x", "y", "z", "yaw", "pitch_sin", "pitch_cos", "roll"])
    coords = []

    affine_mat = data["affine_mat"]
    if isinstance(affine_mat, torch.Tensor):
        affine_mat = affine_mat.data.cpu().numpy()


    inv_camera_mat = np.linalg.inv(load_camera_matrix())
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]["confidence"] = 1 / (1 + np.exp(-logits[r, c]))
        print(type(coords[-1]["x"]), type(coords[-1]["y"]), type(coords[-1]["z"]), type(data["affine_mat"]), type(inv_camera_mat))
        coords[-1]["x"], coords[-1]["y"], coords[-1]["z"] = optimize_xy(
            r, c, coords[-1]["x"], coords[-1]["y"], coords[-1]["z"], affine_mat, inv_camera_mat
        )
    coords = clear_duplicates(coords)
    return coords


def setup_model(model: nn.Module, device, path: Path = None):
    if path is not None:
        model.load_state_dict(torch.load(str(path), map_location=device))
    model.to(device)
