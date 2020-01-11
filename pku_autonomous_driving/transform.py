import numpy as np
import math
import cv2
from typing import Dict
from .geometry import rotate, proj_world_to_screen
from .io import load_camera_matrix

def proj_point(regr_dict, affine_mat):
    world_coords = np.array([
        [regr_dict["x"], regr_dict["y"], regr_dict["z"]],
        [regr_dict["x"], regr_dict["y"] + 0.8, regr_dict["z"]]
    ]).reshape(-1, 3)

    screen_coords = proj_world_to_screen(world_coords)
    screen_coords[:, 2] = 1
    proj_coords = screen_coords[:,[1,0,2]] @ (np.linalg.inv(affine_mat).T)

    y, x = proj_coords[0, 0], proj_coords[0, 1]
    r = proj_coords[1, 0] - y
    return x, y, r

class CropBottomHalf:
    def __init__(self):
        pass

    def __call__(self, input: Dict):
        img, affine_mat = input["img"], input["affine_mat"]

        m = np.array([[1.0, 0, img.shape[0] // 2], [0, 1,  0], [0, 0, 1]], dtype=np.float64)
        affine_mat = m @ affine_mat

        img = img[img.shape[0] // 2:]

        return {**input, "img": img, "affine_mat": affine_mat}


class CropFar:
    def __init__(self, crop_width, crop_height):
        self._crop_bottom_half = CropBottomHalf()
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __call__(self, input: Dict):
        input = self._crop_bottom_half(input)
        img, affine_mat = input["img"], input["affine_mat"]

        hor_offset = max(0, img.shape[1] - self.crop_width) // 2
        m = np.array([[1.0, 0, 0], [0, 1, hor_offset], [0, 0, 1]], dtype=np.float64)
        affine_mat = m @ affine_mat

        img = img[:self.crop_height,hor_offset:-hor_offset]

        return {**input, "img": img, "affine_mat": affine_mat}



class PadByMean:
    def __init__(self, pad_ratio: float=0.25):
        self.pad_ratio = pad_ratio

    def __call__(self, input: Dict):
        img, affine_mat = input["img"], input["affine_mat"]
        pad_width = int(self.pad_ratio * img.shape[1])

        m = np.array([[1.0, 0, 0], [0, 1, -pad_width], [0, 0, 1]], dtype=np.float64)
        affine_mat = np.dot(m, affine_mat)

        bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
        bg = bg[:, : pad_width]
        img = np.concatenate([bg, img, bg], 1)

        return {**input, "img": img, "affine_mat": affine_mat}


class Resize:
    def __init__(self, resized_width, resized_height):
        self.resized_width = resized_width
        self.resized_height = resized_height

    def __call__(self, input: Dict):
        img, affine_mat = input["img"], input["affine_mat"]

        fy = img.shape[0] / self.resized_height
        fx = img.shape[1] / self.resized_width
        m0 = np.array([[1, 0, -affine_mat[0,2]], [0, 1, -affine_mat[1,2]], [0, 0, 1]], dtype=np.float64)
        m1 = np.array([[fy, 0, 0], [0, fx, 0], [0, 0, 1]], dtype=np.float64)
        m2 = np.array([[1, 0, affine_mat[0,2]], [0, 1, affine_mat[1,2]], [0, 0, 1]], dtype=np.float64)
        affine_mat = m2 @ m1 @ m0 @ affine_mat

        img = cv2.resize(img, (self.resized_width, self.resized_height))

        return {**input, "img": img, "affine_mat": affine_mat}


class Normalize:
    def __init__(self):
        pass


    def __call__(self, input: Dict):
        img = input["img"]
        img = (img / 255).astype("float32")

        return {**input, "img": img}


class DropPointsAtOutOfScreen:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def __call__(self, input: Dict):
        data, affine_mat = input["data"], input["affine_mat"]

        valid_regr_dicts = []
        for regr_dict in data:
            x, y, _ = proj_point(regr_dict, affine_mat)
            if (0 <= x < self.screen_width and 0 <= y < self.screen_height):
                valid_regr_dicts.append(regr_dict)
        return {**input, "data": valid_regr_dicts}



class CreateMaskAndRegr:
    def __init__(self, screen_width, screen_height, model_scale):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model_scale = model_scale
        self.inv_camera_matrix = np.linalg.inv(load_camera_matrix())

    def _regr_preprocess(self, regr_dict, regr_x, regr_y, affine_mat, hor_flip):
        proj_coords = np.array([self.model_scale * regr_y, self.model_scale * regr_x, 1])
        est_pos = ((regr_dict["z"] * affine_mat @ proj_coords)[[1, 0, 2]]) @ self.inv_camera_matrix.T
        regr_dict["x"] -= est_pos[0]
        regr_dict["y"] -= est_pos[1]
        regr_dict["z"] /= 100

        regr_dict["roll"] = rotate(regr_dict["roll"], np.pi)
        if hor_flip:
            regr_dict["pitch"] = rotate(regr_dict["pitch"], -2 * regr_dict["pitch"])
        regr_dict["pitch_sin"] = math.sin(regr_dict["pitch"])
        regr_dict["pitch_cos"] = math.cos(regr_dict["pitch"])
        regr_dict.pop("pitch")
        regr_dict.pop("id")
        return regr_dict

    def __call__(self, input: Dict):
        data, affine_mat = input["data"], input["affine_mat"]

        mask_width = self.screen_width // self.model_scale
        mask_height = self.screen_height // self.model_scale
        mesh_x, mesh_y = np.meshgrid(range(mask_width), range(mask_height))

        def _smooth_kernel(x, y, var):
            return np.exp(-(np.square(mesh_x - x) + np.square(mesh_y - y)) / (2 * var))

        def _smooth_regr(regr_dict, x, y, mask):
            points = np.where(0.1 < mask)

            regr = np.zeros([mask.shape[0], mask.shape[1], 7], dtype="float32")
            for py, px in zip(*points):
                regr_dict2 = self._regr_preprocess({**regr_dict}, px, py, affine_mat, False)
                regr[py, px] = np.array([regr_dict2[n] for n in sorted(regr_dict2)])
            return regr

        smooth_masks = []
        smooth_regrs = []
        for regr_dict in data:
            x, y, r = proj_point(regr_dict, affine_mat)
            var = (0.8 * r) / 3.0
            x = np.floor(x / self.model_scale).astype("int")
            y = np.floor(y / self.model_scale).astype("int")
            smooth_masks.append(_smooth_kernel(x, y , var))
            smooth_regrs.append(_smooth_regr(regr_dict, x, y, smooth_masks[-1]))

        mask = np.max(smooth_masks, axis=0)
        regr = np.choose(np.argmax(smooth_masks, axis=0)[:,:,None], smooth_regrs)
        return {**input, "mask": mask, "regr": regr}


class ToCHWOrder:
    def __init__(self):
        pass

    def __call__(self, input: Dict):
        updates = {}
        if "img" in input:
            updates["img"] = np.rollaxis(input["img"], 2, 0)

        if "regr" in input:
            updates["regr"] = np.rollaxis(input["regr"], 2, 0)

        return {**input, **updates}
