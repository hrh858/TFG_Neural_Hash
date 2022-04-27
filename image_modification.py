from statistics import mean
from matplotlib import pyplot as plt
import numpy as np
import cv2

MODIFICATIONS = [
    {
        "method": "rotate",
        "angle": 10
    },
    {
        "method": "rotate",
        "angle": 15
    },
    {
        "method": "rotate",
        "angle": -10
    },
    {
        "method": "rotate",
        "angle": 30
    },
    {
        "method": "rotate",
        "angle": -25
    },
    {
        "method": "rotate",
        "angle": -5
    },
    {
        "method": "rotate",
        "angle": 3
    },
    {
        "method": "rotate",
        "angle": 45
    },
    {
        "method": "blur",
        "size": 1
    },
    {
        "method": "blur",
        "size": 2
    },
    {
        "method": "blur",
        "size": 3
    },
    {
        "method": "blur",
        "size": 4
    },
    {
        "method": "blur",
        "size": 5
    },
    {
        "method": "shift",
        "x": 5,
        "y": 0
    },
    {
        "method": "shift",
        "x": -5,
        "y": 0
    },
    {
        "method": "shift",
        "x": 0,
        "y": 5
    },
    {
        "method": "shift",
        "x": 0,
        "y": -5
    },
    {
        "method": "shift",
        "x": 5,
        "y": 5
    },
    {
        "method": "shift",
        "x": -5,
        "y": -5
    },
    {
        "method": "shift",
        "x": -5,
        "y": 5
    },
    {
        "method": "shift",
        "x": 5,
        "y": -5
    },
]

def generate_modifications(images, transformations):
    generated_images = np.ndarray((images.shape[0], len(transformations), images.shape[1], images.shape[1]))

    for i in range(len(images)):
        original_image = images[i, :, :]
        modifications = []
        for transformation in transformations:
            method = transformation['method']

            if method == "rotate":
                angle = transformation["angle"]
                image_center = tuple(np.array(original_image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                generated_image = cv2.warpAffine(original_image, rot_mat, original_image.shape[1::-1], flags=cv2.INTER_LINEAR)
            elif method == "blur":
                size = transformation["size"]
                ksize = (size, size)
                generated_image = cv2.blur(original_image, ksize)
            elif method == "shift":
                x = transformation["x"]
                y = transformation["y"]
                m = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
                rows, cols = original_image.shape
                generated_image = cv2.warpAffine(original_image, m, (cols, rows))
            else:
                print(f'Tried to apply an unsupported transformation method "{method}"')
                raise
                
            modifications.append(generated_image)
        
        generated_images[i] = np.array(modifications, dtype=np.uint8)
    
    return generated_images
        
def save_modifications(path, modifications):
    np.save(path, modifications)

def load_modifications(path):
    return np.load(path)