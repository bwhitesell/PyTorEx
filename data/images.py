import numpy as np
from matplotlib import pyplot as plt


def show_tensor(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')