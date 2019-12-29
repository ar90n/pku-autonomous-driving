import numpy as np

from .io import load_camera_matrix


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


def get_img_coords(s):
    """
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    """
    coords = str2coords(s)
    xs = [c["x"] for c in coords]
    ys = [c["y"] for c in coords]
    zs = [c["z"] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T

    camera_matrix = load_camera_matrix()
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2]  # z = Distance from the camera
    print(np.dot(camera_matrix, P).T)
    return img_xs, img_ys
