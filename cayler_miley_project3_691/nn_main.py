import numpy as np
from sklearn.datasets import make_moons, make_blobs
from matplotlib import pyplot as plt

import nn

GRAD_FLAG = True

if __name__ == "__main__":
    np.random.seed(0)

    if GRAD_FLAG:
        X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=0)
        outputs = np.ptp(y) + 1
    else:
        X, y = make_moons(200, noise=0.2)
        outputs = np.ptp(y) + 1
    plt.scatter(X[:, 0], X[:, 1], s=40, cmap=plt.cm.get_cmap('Spectral'), c=y)

    plt.figure(figsize=(16, 16))
    hidden_layer_dimensions = [1, 2, 3, 4]

    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(2, 2, i+1)
        plt.title('HiddenLayerSize{}'.format(nn_hdim))
        model = nn.build_model(X, y, nn_hdim, print_loss=False, outputs=outputs)
        nn.plot_decision_boundary(lambda x: nn.predict(model, x), X, y)

    plt.show()
