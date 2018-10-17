import perceptron as pe
import gradient_descent as gd
import numpy as np
import math

from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import matplotlib.pyplot as plt
from matplotlib import cm

X1 = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
Y1 = np.array([[1], [1], [-1], [1], [-1], [-1], [-1]])

X2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y2 = np.array([[1], [1], [1], [-1], [-1], [-1]])


def fun(x, y):
    return x ** 2 + y


def df_squared(x):
    return np.array([2 * x[0]])


def best_grad(x):
    a, b = 10, 7
    return np.array([2.0 * (x[0] - a), 2.0 * (x[1] - b)])


def df(x):
    return np.array([4 * math.pow(x[0], 3), 6 * x[1]])


def main():
    w1 = pe.perceptron_train(X1, Y1, True)
    pe.perceptron_test(X1, Y1, w1[0], w1[1])

    w2 = pe.perceptron_train(X2, Y2, True)
    pe.perceptron_test(X1, Y1, w2[0], w2[1], True)

    x = gd.gradient_descent(df_squared, np.array([5.0]), 0.1)
    print("minimum occurs at: {}".format(x))
    x2 = gd.gradient_descent(df, np.array([1.0, 1.0]), 0.01)
    print("minimum occurs at: {}".format(x2))
    x3 = gd.gradient_descent(best_grad, np.zeros(2), 0.01)
    print("minimum occurs at: {}".format(x3))

    plt.style.use('ggplot')

    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    main()
