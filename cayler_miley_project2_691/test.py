import perceptron as pe
import gradient_descent as gd
import numpy as np
import math

from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import matplotlib.pyplot as plt
import matplotlib

X1 = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
Y1 = np.array([[1], [1], [-1], [1], [-1], [-1], [-1]])

X2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y2 = np.array([[1], [1], [1], [-1], [-1], [-1]])


def fun(x, y):
    return x ** 2 + y


def df_squared(x):
    return np.array([2*x[0], 2*x[1]])


def best_grad(x):
    a, b = 10, 7
    return np.array([2.0 * (x[0] - a), 2.0 * (x[1] - b)])


def df(x):
    return np.array([4 * math.pow(x[0], 3), 6 * x[1]])


def decision(val):
    return -0.5*val - 0.5


def main():
    w1 = pe.perceptron_train(X1, Y1)
    pe.perceptron_test(X1, Y1, w1[0], w1[1])

    w2 = pe.perceptron_train(X2, Y2)
    print(w2)
    pe.perceptron_test(X1, Y1, w2[0], w2[1])

    x = gd.gradient_descent(df_squared, np.array([5.0, 5.0]), 0.1)
    print("minimum occurs at: {}".format(x))
    # x2 = gd.gradient_descent(df, np.array([1.0, 1.0]), 0.01)
    # print("minimum occurs at: {}".format(x2))
    # x3 = gd.gradient_descent(best_grad, np.zeros(2), 0.01)
    # print("minimum occurs at: {}".format(x3))

    plt.style.use('ggplot')

    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    indep = X2[:]
    depen = [decision(value) for value in indep]

    plt.title("Perceptron Decision Boundary on Sample Data")
    plt.scatter(X2[:3, 0], X2[:3, 1], c='g', marker='+', label="Positive")
    plt.scatter(X2[3:, 0], X2[3:, 1], c='r', marker='.', label="Negative")
    plt.plot(indep, depen, c='b')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc=9)

    plt.savefig('percep_decision.png')


if __name__ == "__main__":
    main()
