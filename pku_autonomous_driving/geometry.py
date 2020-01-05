from math import sin, cos

import numpy as np
from .io import load_camera_matrix


def proj_world_to_screen(world_coords, camera_matrix=None):
    """
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    """
    if camera_matrix is None:
        camera_matrix = load_camera_matrix()

    img_p = np.dot(world_coords, camera_matrix.T)
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    return img_p


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))
