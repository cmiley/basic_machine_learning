import numpy as np
import math


def gradient_descent(grad, x_init, eta):
    # use stopping point for gradient value sufficiently close to 0
    threshold = 1e-5

    # use the initial values as the minimum before descending
    minimum = x_init

    # while the gradient is effectively non-zero
    while np.linalg.norm(grad(minimum)) > threshold:
        # min = min - eta*grad
        minimum = np.subtract(minimum, np.multiply(eta, grad(minimum)))

    return minimum


def df_squared(x):
    return np.array([2 * x[0]])


def best_grad(x):
    a, b = 10, 7
    return np.array([2.0 * (x[0] - a), 2.0 * (x[1] - b)])


def df(x):
    return np.array([4 * math.pow(x[0], 3), 6 * x[1]])


def main():
    x = gradient_descent(df_squared, np.array([5.0]), 0.1)
    # print("minimum occurs at: {}".format(x))
    x2 = gradient_descent(df, np.array([1.0, 1.0]), 0.01)
    # print("minimum occurs at: {}".format(x2))
    x3 = gradient_descent(best_grad, np.zeros(2), 0.01)
    # print("minimum occurs at: {}".format(x3))


if __name__ == "__main__":
    main()
