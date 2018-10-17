import numpy as np

X1 = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
Y1 = np.array([[1], [1], [-1], [1], [-1], [-1], [-1]])

X2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y2 = np.array([[1], [1], [1], [-1], [-1], [-1]])


def update_weights(sample, label, weights, bias, FLAG_debug=False):
    activation = np.dot(sample, weights) + bias
    if activation * label <= 0:
        for index, weight in enumerate(weights):
            weights[index] = weight + sample[index] * label

        bias = bias + label

    if FLAG_debug:
        print("W: {}, B: {}".format(weights, bias))

    return weights, bias


def perceptron_train(X, Y, FLAG_debug=False):
    # initialize weights and bias
    w = np.zeros(X[0].size)
    b = 0
    max_epochs = 1000

    w_stale = np.zeros(X[0].size)
    b_stale = 0

    for epoch in range(max_epochs):
        w_update = []
        for index, sample in enumerate(X):
            w, b = update_weights(sample, Y[index], w, b)
            if np.array_equal(w_stale, w) and b_stale == b:
                w_update.append(True)
            else:
                w_update.append(False)
            w_stale = w
            b_stale = b
        if np.all(w_update):
            if FLAG_debug:
                print("Converged at epoch: {}".format(epoch))
            break
        else:
            if FLAG_debug:
                print("Not converged at epoch: {}".format(epoch))

    if FLAG_debug:
        print("W: {}, B: {}".format(w, b))

    return [w, b]


def perceptron_test(X_test, Y_test, w, b, FLAG_debug=False):
    correct = 0

    for index, sample in enumerate(X_test):
        activation = np.dot(sample, w) + b
        if activation * Y_test[index] > 0:
            correct += 1

    accuracy = correct / len(X_test)

    if FLAG_debug:
        print("Number correct: {} out of {}".format(correct, len(X_test)))
        print("Testing accuracy: {}".format(accuracy))

    return accuracy


def main():
    w1 = perceptron_train(X1, Y1)
    perceptron_test(X1, Y1, w1[0], w1[1])

    w2 = perceptron_train(X2, Y2)
    perceptron_test(X1, Y1, w2[0], w2[1])


if __name__ == "__main__":
    main()
