from model import siamese_nn_model_basiccnn
from loss import contrastive_loss
from data_loaders import load_mnist
from image_modification import MODIFICATIONS, generate_modifications, save_modifications, load_modifications
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--load_train', action=argparse.BooleanOptionalAction)
parser.add_argument('--show_model_summary', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

network = siamese_nn_model_basiccnn((28, 28, 1))
if args.show_model_summary: network.summary()

images, labels = load_mnist()

if args.load_train:
    modified_images = load_modifications('data/mnist/generated_modifications.npy')
else:
    modified_images = generate_modifications(images, MODIFICATIONS) 
    save_modifications('data/mnist/generated_modifications.npy', modified_images)