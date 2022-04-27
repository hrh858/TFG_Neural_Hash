import numpy as np

def generate_pairs(original_images, modified_images):
    positive_pairs = []
    negative_pairs = []

    for img_idx in range(original_images.shape[0]):
        original_image = original_images[img_idx]
        for modified_image in modified_images[img_idx]:
            positive_pairs.append([original_image, modified_image])

            distinct_img_idx = -1
            while distinct_img_idx < 0 or distinct_img_idx == img_idx: 
                distinct_img_idx = np.random.randint(0, original_images.shape[0])
            distinct_image = original_images[distinct_img_idx]
            negative_pairs.append([original_image, distinct_image])

    positive_pairs_arr = np.array(positive_pairs, dtype=np.float64) / 255.0
    negative_pairs_arr = np.array(negative_pairs, dtype=np.float64) / 255.0

    return positive_pairs_arr, negative_pairs_arr