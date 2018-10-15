import numpy as np


def gradient_descent(grad, x_init, eta):
    threshold = 1e-5

    minimum = x_init
    while grad(minimum) > threshold:
        minimum = minimum - eta*grad(minimum)

    return minimum


def main():
    pass


if __name__ == "__main__":
    main()