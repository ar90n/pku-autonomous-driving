import numpy as np
import cv2
from .geometry import euler_to_rot_mat, proj_world_to_screen
from .io import load_camera_matrix


def rect_of_car():
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    P = np.array(
        [
            [x_l, -y_l, -z_l, 1],
            [x_l, -y_l, z_l, 1],
            [-x_l, -y_l, z_l, 1],
            [-x_l, -y_l, -z_l, 1],
            [x_l, y_l, -z_l, 1],
            [x_l, y_l, z_l, 1],
            [-x_l, y_l, z_l, 1],
            [-x_l, y_l, -z_l, 1],
            [0, 0, 0, 1],
        ]
    ).T
    return P


def translate_mat(x, y, z, yaw, pitch, roll):
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_rot_mat(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    return Rt


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[4][:2]), tuple(points[7][:2]), color, 16)
    cv2.line(image, tuple(points[4][:2]), tuple(points[5][:2]), color, 16)
    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 16)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        p_z = max(p_z, 1.0)
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    #         if p_x > image.shape[1] or p_y > image.shape[0]:
    #             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image


def draw_coords(img, coords):
    # You will also need functions from the previous cells

    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point["x"], point["y"], point["z"]
        yaw, pitch, roll = -point["pitch"], -point["yaw"], -point["roll"]

        # Math
        Rt = translate_mat(x, y, z, yaw, pitch, roll)
        P = rect_of_car()

        img_cor_points = proj_world_to_screen(np.dot(Rt, P).T).astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])

    return img
