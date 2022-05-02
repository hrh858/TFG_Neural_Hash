import numpy as np
import cv2


def image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_image(image):
    return image / 255.0

def n_random_numbers(n, max, min=0):
    numbers = []
    while len(numbers) < n:
        rand_num = np.random.randint(min, max, 1)[0]
        if rand_num not in numbers:
            numbers.append(rand_num)
    return numbers

def supplementary_numbers(source, max):
    numbers = []
    for i in range(max):
        if i not in source:
            numbers.append(i)
    return numbers