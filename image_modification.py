from flask import blueprints
import numpy as np
import cv2

def generate_modifications_map(n_rotations, n_blurs, n_shifts, n_patches):

    modifications = []

    rotation_values = np.random.randint(-90, 90+1, n_rotations)
    blur_values = np.random.randint(1, 10+1, n_blurs)
    shifts_x = np.random.randint(-10, 10+1, n_shifts)
    shifts_y = np.random.randint(-10, 10+1, n_shifts)
    start_points_x = np.random.randint(5, 27+1, n_patches)
    start_points_y = np.random.randint(5, 27+1, n_patches)
    patch_sizes = np.random.randint(5, 15+1, n_patches)

    for rot in rotation_values:
        modifications.append({
            "method": "rotate",
            "angle": int(rot)
        })
    for blur in blur_values:
        modifications.append({
            "method": "blur",
            "size": int(blur)
        })
    for shift_x, shift_y in zip(shifts_x, shifts_y):
        modifications.append({
            "method": "shift",
            "x": int(shift_x),
            "y": int(shift_y)
        })
    for starting_point_x, starting_point_y, patch_size in zip(start_points_x, start_points_y, patch_sizes):
        modifications.append({
            "method": "patch",
            "start_point": (int(starting_point_x), int(starting_point_y)),
            "end_point": (int(starting_point_x + patch_size), int(starting_point_y + patch_size))
        })

    return modifications


def modify_image(image, transformation):
    method = transformation["method"]
    
    if method == "rotate":
        angle = transformation["angle"]
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        generated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    elif method == "blur":
        size = transformation["size"]
        ksize = (size, size)
        generated_image = cv2.blur(image, ksize)
    elif method == "shift":
        x = transformation["x"]
        y = transformation["y"]
        m = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
        rows, cols = image.shape
        generated_image = cv2.warpAffine(image, m, (cols, rows))
    elif method == "patch":
        start_point = transformation["start_point"]
        end_point = transformation["end_point"]
        generated_image = cv2.rectangle(image.copy(), start_point, end_point, (0, 0, 0), -1)
    else:
        print(f'Tried to apply an unsupported transformation method "{method}"')
        raise
    return generated_image