import math
from math import sin, cos, sqrt, atan2

import numpy as np
from .io import load_camera_matrix
from cv2 import Rodrigues


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
def euler_to_rot_mat(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    #R0 = np.dot(Y, np.dot(P, R))
    #R1 = np.dot(R, np.dot(Y, P))
    #print(vv(R0))
    #print(vv(R1))
    #print(d(R0))
    #print(d(R1))
    #return np.dot(Y, np.dot(P, R))
    return np.dot(R, np.dot(Y, P))

def vv(R):
    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * math.sin(theta))
    return theta, multi

def d(R):
    print((R[2, 1] , R[1, 2]), (R[0, 2] , R[2, 0]), (R[1, 0] , R[0, 1]))
    return (R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])

def euler_to_rot_vec(yaw, pitch, roll):
    R = euler_to_rot_mat(yaw, pitch, roll)

    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * math.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    return np.array([rx, ry, rz])
#
#    print(rot_mat)
#    rot_vec = Rodrigues(rot_mat)[0]
#    rot_vec = np.squeeze(rot_vec)
#    return rot_vec

def rot_vec_to_euler(rot_vec):
    # Rotation Vector -> Rotation Matrix
    print(rot_vec)
    R = Rodrigues(rot_vec)[0]
    print(R)

    sq = sqrt(R[1,0] ** 2 +  R[1,1] ** 2)
    print(sq)

    if  not (sq < 1e-6) :
        pitch = atan2(-R[1,2], sq)
        print(math.pi <= abs(2 * pitch))
        print(2 * pitch)
        print(sq)
        yaw = atan2(R[0,2] , R[2,2])
        roll = atan2(R[1,0], R[1,1])
    else :
        yaw = 0
        pitch = atan2(-R[1,2], sq)
        roll = atan2(-R[1,2] * R[2,0], -R[1,2]* R[2,1])

    return yaw, pitch, roll



def _clamp_radian(theta, minv=0, maxv = (2 * math.pi)):
    if theta < minv:
        return _clamp_radian(theta + 2 * math.pi, minv, maxv)
    elif maxv <= theta:
        return _clamp_radian(theta - 2 * math.pi, minv, maxv)
    return theta

def calc_global_pitch(org_pitch):
    return _clamp_radian(org_pitch + math.pi / 2)

def calc_org_pitch(global_pitch):
    return _clamp_radian(global_pitch - math.pi / 2, -math.pi, math.pi)

def calc_ray_pitch(x, camera_matrix=load_camera_matrix()):
    diff = x - camera_matrix[0, 2]
    ray_pitch = math.pi / 2 + math.atan(diff / camera_matrix[0,0])
    return ray_pitch
