import numpy as np

X2 = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y2 = np.array([[1], [1], [1], [-1], [-1], [-1]])


def update_weights(sample, label, weights, bias, FLAG_debug=False):
    activation = np.dot(sample, weights) + bias

    # if prediction is wrong
    if activation * label <= 0:
        # update each weight
        for index, weight in enumerate(weights):
            weights[index] = weight + sample[index] * label

        # update bias
        bias = bias + label

    if FLAG_debug:
        print("W: {}, B: {}".format(weights, bias))

    return weights, bias


def perceptron_train(X, Y, FLAG_debug=False):
    # initialize weights and bias to zero
    w = np.zeros(X[0].size)
    b = 0

    # set limit for non-linearly separable data
    max_epochs = 1000

    w_stale = np.zeros(X[0].size)
    b_stale = 0

    for epoch in range(max_epochs):
        w_update = []
        for index, sample in enumerate(X):
            w, b = update_weights(sample, Y[index], w, b)

            # append to list of T/F for weight and bias changes
            if np.array_equal(w_stale, w) and b_stale == b:
                w_update.append(True)
            else:
                w_update.append(False)
            w_stale = w
            b_stale = b

        # if weights have not changed for an entire epoch
        if np.all(w_update):
            if FLAG_debug:
                print("Converged at epoch: {}".format(epoch))

            # break and return
            break
        else:
            if FLAG_debug:
                print("Not converged at epoch: {}".format(epoch))

    if FLAG_debug:
        print("W: {}, B: {}".format(w, b))

    return [w, b]


def perceptron_test(X_test, Y_test, w, b, FLAG_debug=False):
    correct = 0

    # for each sample in the test set, compare the prediction with the label
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
    w2 = perceptron_train(X2, Y2, True)
    perceptron_test(X2, Y2, w2[0], w2[1], True)


if __name__ == "__main__":
    main()
