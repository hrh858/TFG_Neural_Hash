import numpy as np
import pdb

from image_modification import modify_image

def generate_transformations(images, transformations):
    transformed_images = []
    for image in images:
        t = []
        for transformation in transformations:
            t.append(modify_image(image, transformation))
        transformed_images.append(t)

def generate_pairs(original_images, transformed_images):
    positive_pairs = []
    negative_pairs = []

    for idx in range(original_images.shape[0]):
        or_im = original_images[idx]
        trans_ims = transformed_images[idx]
        for trans_im in trans_ims:

            rndm_idx = np.random.randint(0, original_images.shape[0])
            while rndm_idx == idx: 
                rndm_idx = np.random.randint(0, original_images.shape[0])
            rndm_im_for_neg_pair = original_images[rndm_idx]

            positive_pairs.append([or_im, trans_im])
            negative_pairs.append([or_im, rndm_im_for_neg_pair])

    pdb.set_trace()

    return (
        np.array(positive_pairs, dtype=np.float64),
        np.array(negative_pairs, dtype=np.float64)
    )