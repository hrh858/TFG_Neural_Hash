from generate_pairs import generate_pairs
from utils import image_to_grayscale, n_random_numbers, normalize_image, supplementary_numbers
from image_modification import generate_modifications_map, modify_image
from distance import euclidean_distance
from model import siamese_nn_model_basiccnn
from loss import contrastive_loss
from data_loaders import load_mnist
from plots import plot_curves, plot_pairs
import numpy as np
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets.cifar10 import load_data
import argparse
import tensorflow as tf
import pdb

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

parser = argparse.ArgumentParser()
parser.add_argument('--load_train', action=argparse.BooleanOptionalAction)
parser.add_argument('--show_model_summary', action=argparse.BooleanOptionalAction)
parser.add_argument('--plot_pairs', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

MODIFICATIONS = generate_modifications_map(100, 100, 100, 100)
N_MODIFICATIONS = len(MODIFICATIONS)

# images, labels = load_mnist()
(images_train, _), (images_val,_) = load_data()
images = np.concatenate((images_train[:100], images_val[:10]))
print(images.shape)

images = map(image_to_grayscale, images)
images = map(normalize_image, images)
images = list(images)

modified_images = []
for image in images:
    m = []
    for modification in MODIFICATIONS:
        m.append(modify_image(image, modification))
    modified_images.append(m)

original_images = np.array(images, dtype=np.float64)
modified_images = np.array(modified_images, dtype=np.float64)

modified_images_train = []
modified_images_test = []
for idx in range(len(modified_images)):
    train_idxs = n_random_numbers(300, max=N_MODIFICATIONS, min=0)
    test_idxs = supplementary_numbers(train_idxs, max=N_MODIFICATIONS)
    modified_images_train.append(modified_images[idx][train_idxs])
    modified_images_test.append(modified_images[idx][test_idxs])
modified_images_train = np.array(modified_images_train)
modified_images_test = np.array(modified_images_test)

pos_train_pairs, neg_train_pairs = generate_pairs(original_images, modified_images_train)
pos_test_pairs, neg_test_pairs = generate_pairs(original_images, modified_images_test)

if args.plot_pairs:
    plot_pairs(pos_train_pairs[0:15])
    plot_pairs(neg_train_pairs[0:15])

x_train = np.expand_dims(np.concatenate((pos_train_pairs, neg_train_pairs)), axis=4)
y_train = np.concatenate((np.full(pos_train_pairs.shape[0], True), np.full(neg_train_pairs.shape[0], False)))

x_test = np.expand_dims(np.concatenate((pos_test_pairs, neg_test_pairs)), axis=4)
y_test = np.concatenate((np.full(pos_test_pairs.shape[0], 1.0), np.full(neg_test_pairs.shape[0], 0.0)))


pair_image_a = Input(shape=(32, 32, 1))
pair_image_b = Input(shape=(32, 32, 1))

feature_extractor = siamese_nn_model_basiccnn((32, 32, 1))
if args.show_model_summary: feature_extractor.summary()
features_image_a = feature_extractor(pair_image_a)
features_image_b = feature_extractor(pair_image_b)
distance = Lambda(euclidean_distance)([features_image_a, features_image_b])
model = Model(inputs=[pair_image_a, pair_image_b], outputs=distance)

print(f'''Train shape: {x_train.shape}''')
print(f'''Test shape: {x_test.shape}''')

model.compile(loss=contrastive_loss, optimizer="adam", metrics=["accuracy"])
with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    history = model.fit(
        [x_train[:,0], x_train[:,1]], y_train[:],
        validation_data=([x_test[:,0], x_test[:,1]], y_test[:]),
        batch_size=64,
        epochs=50,
        workers=12,
    )

plot_curves(history.history['loss'], history.history['val_loss'], 'Loss')
plot_curves(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy')

model.save("output/contrastive_siamese_model")