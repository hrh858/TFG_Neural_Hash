import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_mnist(path="data/mnist/train.csv"):
    df = pd.read_csv(path)
    labels = np.array(df.iloc[:,0], dtype=np.uint8)
    images = np.array([np.array(df.iloc[i, 1:], dtype=np.uint8).reshape((28, 28)) for i in range(len(labels))])
    return images, labels