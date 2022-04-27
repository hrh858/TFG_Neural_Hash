import numpy as np
import matplotlib.pyplot as plt

def plot_pairs(positive_pairs, negative_pairs, n=1):
    for i in range(n):
        idx = np.random.randint(0, positive_pairs.shape[0])
        positive_pair = positive_pairs[idx]
        negative_pair = negative_pairs[idx]

        w = positive_pair.shape[0]
        h = positive_pair.shape[1]
        fig = plt.figure(figsize=(2,2))
        for i in range(1, 5):
            fig.add_subplot(2, 2, i)
            if i == 1:
                img = positive_pair[0]
            elif i == 2:
                img = positive_pair[1]
            elif i == 3:
                img = negative_pair[0]
            elif i == 4:
                img = negative_pair[1]
            plt.imshow(img)
        plt.show()
