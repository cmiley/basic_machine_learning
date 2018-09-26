import numpy as np

## K-NN Data
X_train1 = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1],
                     [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
Y_train1 = np.array([[-1], [-1], [1], [-1], [1], [-1], [1],
                     [-1], [1], [-1], [1], [-1], [1], [1]])

X_test1 = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5],
                    [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y_test1 = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

'''
Basic Requirements: 
'''


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    correct = 0.0

    for index, point in enumerate(X_test):
        label = test_point(X_train, Y_train, point, K)

        if label == Y_test[index][0]:
            correct += 1

    accuracy = correct / len(X_test)
    print(accuracy)

    return accuracy


'''
691 (Graduate) Requirements:
'''


def choose_K(X_train, Y_train, X_val, Y_val):
    K = len(X_train)
    accuracy = 0

    # for all possible K, check accuracy
    for test_K in range(1, K + 1):
        new_accuracy = KNN_test(X_train, Y_train, X_val, Y_val, test_K)
        if new_accuracy > accuracy:
            accuracy = new_accuracy
            K = test_K

    print(K)
    return K


# Helper functions
def test_point(X_train, Y_train, ref_point, K):
    distances = []

    for index, point in enumerate(X_train):
        distances.append((L2_norm(point, ref_point), Y_train[index][0]))

    sorted_dist = sorted(distances, key=lambda item: item[0])[:K]

    output_label = sum([pair[1] for pair in sorted_dist])

    if output_label == 0:
        output_label = sorted_dist[0][1]
    output_label = np.sign(output_label)

    return output_label


def L2_norm(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def main():
    KNN_test(X_train1, Y_train1, X_test1, Y_test1, 3)
    choose_K(X_train1, Y_train1, X_test1, Y_test1)


if __name__ == '__main__':
    main()
