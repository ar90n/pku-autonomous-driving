import numpy as np
from scipy.optimize import minimize

from .const import DISTANCE_THRESH_CLEAR
from .geometry import convert_3d_to_2d, proj_world_to_screen, rotate

# from .improc import regr_back


def regr_back(regr_dict):
    for name in ["x", "y", "z"]:
        regr_dict[name] = regr_dict[name] * 100
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
    return proj_world_to_screen(coords)


def optimize_xy(r, c, x0, y0, z0):
    def distance_fn(xyz):
        x, y, z = xyz
        x, y = convert_3d_to_2d(x, y, z0)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        x = np.round(x).astype("int")
        y = (y + IMG_SHAPE[1] // 4) * IMG_WIDTH / (IMG_SHAPE[1] * 1.5) / MODEL_SCALE
        y = np.round(y).astype("int")
        return (x - r) ** 2 + (y - c) ** 2

    res = minimize(distance_fn, [x0, y0, z0], method="Powell")
    x_new, y_new, z_new = res.x
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


def extract_coords(prediction):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(["x", "y", "z", "yaw", "pitch_sin", "pitch_cos", "roll"])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(regr_back(regr_dict))
        coords[-1]["confidence"] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]["x"], coords[-1]["y"], coords[-1]["z"] = optimize_xy(
            r, c, coords[-1]["x"], coords[-1]["y"], coords[-1]["z"]
        )
    coords = clear_duplicates(coords)
    return coords
