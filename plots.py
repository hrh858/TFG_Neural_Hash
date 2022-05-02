import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_pairs(pairs):
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(len(pairs), 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for idx in range(len(pairs)):
        im1, im2 = pairs[idx]
        grid[idx*2].imshow(im1)
        grid[(idx*2)+1].imshow(im2)
    plt.show()

def plot_curves(train, test, title):
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, 'b', label="Train")
    plt.plot(epochs, test, 'orange', label="Validation")
    plt.title(title)
    plt.legend()
    plt.show()
