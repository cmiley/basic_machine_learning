import numpy as np


def update_weights(sample, label, weights, bias, FLAG_debug=False):
    activation = np.dot(sample, weights) + bias
    if activation*label <= 0:
        for index, weight in enumerate(weights):
            weights[index] = weight + sample[index]*label

        bias = bias + label

    if FLAG_debug:
        print("W: {}, B: {}".format(weights, bias))

    return weights, bias


def perceptron_train(X, Y, FLAG_debug=False):
    # initialize weights and bias
    w = np.zeros(X[0].size)
    b = 0

    for index, sample in enumerate(X):
        w, b = update_weights(sample, Y[index], w, b)

    if FLAG_debug:
        print("Debuggin...")
        print("W: {}, B: {}".format(w, b))

    return w, b


def perceptron_test(X_test, Y_test, w, b, FLAG_debug=False):
    correct = 0

    for index, sample in enumerate(X_test):
        activation = np.dot(sample, w) + b
        if activation*Y_test[index] > 0:
            correct += 1

    accuracy = correct/X_test.size

    if FLAG_debug:
        print("Testing accuracy: {}".format(accuracy))

    return accuracy


def main():
    pass


if __name__ == "__main__":
    main()