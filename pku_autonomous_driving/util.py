import torch
from torch import nn
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from .const import IMG_WIDTH, IMG_HEIGHT, MODEL_SCALE
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
    est_pos = ((z0 * affine_mat @ proj_coords)[[1, 0, 2]]) @ inv_camera_mat.T
    x_new = x0 + est_pos[0]
    y_new = y0 + est_pos[1]
    return x_new, y_new, z0


def extract_coords(data, prediction=None):
    if prediction is None:
        logits = data["mask"]
        regr_output = data["regr"]
    else:
        logits = prediction[0]
        regr_output = prediction[1:]

    peaks = np.zeros(logits.shape, dtype=np.int)
    peaks[:, 1:] += logits[:, 1:] > logits[:, :-1]
    peaks[1:, :] += logits[1:, :] > logits[:-1, :]
    peaks[:, :-1] += logits[:, :-1] > logits[:, 1:]
    peaks[:-1, :] += logits[:-1, :] > logits[1:,:]
    peaks[:, :] += 0 < logits
    points = np.argwhere(peaks == 5)

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
        coords[-1]["x"], coords[-1]["y"], coords[-1]["z"] = optimize_xy(
            r, c, coords[-1]["x"], coords[-1]["y"], coords[-1]["z"], affine_mat, inv_camera_mat
        )
    return coords


def setup_model(model: nn.Module, device, path: Path = None):
    if path is not None:
        model.load_state_dict(torch.load(str(path), map_location=device)["model"],  strict=False)
    model.to(device)


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))
