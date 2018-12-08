import numpy as np
from matplotlib import pyplot as plt


def calculate_loss(model, X, y):
    y_hat = forward(model, X)
    labels = one_hot_encode(y)

    loss = -(1/y.size)*np.sum(labels*np.log(y_hat))

    return loss


def predict(model, x):
    y_hat = forward(model, x)

    prediction = np.argmax(y_hat, axis=1)

    return np.array(prediction)


def forward(model, X):
    a = X @ model['W1'] + model['b1']
    h = np.tanh(a)
    z = h @ model['W2'] + model['b2']
    y_hat = softmax(z)

    return y_hat


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False, outputs=2, learning_rate=0.002):
    # Randomly assign weights and biases
    model = {
        'W1': np.random.rand(X[0].size, nn_hdim),
        'W2': np.random.rand(nn_hdim, outputs),
        'b1': np.random.rand(nn_hdim),
        'b2': np.random.rand(outputs)
    }

    # modify labels to be one-hot encoded
    labels = one_hot_encode(y)

    for index in range(num_passes):
        # generate vector of predictions
        y_hat = forward(model, X)

        # compute gradients
        h = np.tanh(X @ model['W1'] + model['b1'])

        dLdy_hat = y_hat - labels
        dLda = (1 - h**2) * (dLdy_hat @ model['W2'].T)
        dLdW2 = h.T @ dLdy_hat
        dLdb2 = np.sum(dLdy_hat)
        dLdW1 = X.T @ dLda
        dLdb1 = np.sum(dLda)

        # update weights and biases
        model['W1'] -= learning_rate * dLdW1
        model['W2'] -= learning_rate * dLdW2
        model['b1'] -= learning_rate * dLdb1
        model['b2'] -= learning_rate * dLdb2

        # print loss every thousand iterations (if wanted)
        if print_loss and index % 1000 == 0:
            print(calculate_loss(model, X, y))

    return model


def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    return build_model(X, y, nn_hdim, num_passes, print_loss, outputs=3)


def softmax(z_vec):
    y_hat = np.exp(z_vec)/(np.sum(np.exp(z_vec), axis=1)).reshape(-1, 1)

    return y_hat


def one_hot_encode(vec):

    labels = np.zeros((vec.size, vec.max() - vec.min() + 1))
    for index, label in enumerate(vec):
        labels[index, label] = 1

    return labels


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('Spectral'))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('Spectral'))
