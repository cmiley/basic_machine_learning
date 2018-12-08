import os
from pca import *
from matplotlib import pyplot as plt
import numpy as np


def compress_images(DATA, k):
    if not os.path.isdir('Output'):
        os.mkdir('Output')

    Z_mat = compute_Z(DATA)
    cov_mat = compute_covariance_matrix(Z_mat)
    eigenvals, eigenvecs = find_pcs(cov_mat)
    Z_star = project_data(Z_mat, eigenvecs, eigenvals, k=k, var=0)

    compressed = Z_star @ eigenvecs[:, :k].T

    for index, img in enumerate(compressed.T):
        plt.imsave('Output/{}.png'.format(index), img.reshape(-1, 48), cmap='gray')

    return compressed


def load_data(input_dir):
    img_data = []

    for root, dirs, files in os.walk(input_dir):
        for file in np.sort(files):
            img = plt.imread(input_dir + file, 'pgm')

            # flatten each image
            img_data.append(img.reshape(-1))

    # return flattened images as columns
    return np.array(img_data, dtype=float).T
