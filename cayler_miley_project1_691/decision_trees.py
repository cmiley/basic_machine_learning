import numpy as np

## Decision Trees
# Training Set 1:
X_train_1 = np.array([[0, 1], [0, 0], [1, 0], [0, 0], [1, 1]])
Y_train_1 = np.array([[1], [0], [0], [0], [1]])
# Validation Set 1:
X_val_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_val_1 = np.array([[0], [1], [0], [1]])
# Testing Set 1:
X_test_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_test_1 = np.array([[1], [1], [0], [1]])
# Training Set 2:
X_train_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1],
                      [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_train_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
# Validation Set 2:
X_val_2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1],
                    [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
# Testing Set 2:
X_test_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1],
                     [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

X_real = np.array([[4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 1.2], [5.0, 3.4, 1.6, 0.2],
                   [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
                   [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7.0, 3.2, 4.7, 1.4],
                   [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3],
                   [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6],
                   [4.9, 2.4, 3.3, 1.0]])
Y_real = np.array([[1], [1], [1], [1], [1], [1], [1], [1],
                   [0], [0], [0], [0], [0], [0], [0], [0]])

'''
Basic Requirements:
'''


def DT_train_binary(X, Y, max_depth):
    depth_count = 0

    if max_depth < 0:
        # iterate until maximum accuracy reached
        max_depth = min(len(X[0]), len(X) - 1)

    used_features = []

    tree = choose_best_tree(X, Y, used_features, depth_count, max_depth)

    print(tree)
    return tree


def DT_test_binary(X, Y, DT):
    num_correct = 0

    for s_index, sample in enumerate(X):
        pred = make_prediction(sample, DT)
        if pred == Y[s_index]:
            num_correct += 1

    return num_correct/len(X)


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
    max_depth = min(len(X_train[0]), len(X_train)-1)
    best_acc = 0

    for depth in range(max_depth):
        tree = DT_train_binary(X_train, Y_train, depth)
        accuracy = DT_test_binary(X_val, Y_val, tree)
        if accuracy > best_acc:
            output_tree = tree
            best_acc = accuracy

    return output_tree


'''
691 (Graduate) Requirements:
'''


def DT_train_real(X, Y, max_depth):
    depth_count = 0

    if max_depth < 0:
        # iterate until maximum accuracy reached
        max_depth = min(len(X[0]), len(X) - 1)

    used_features = []

    tree = choose_best_tree(X, Y, used_features, depth_count, max_depth, is_binary=False)

    print(tree)
    return tree


def DT_test_real(X, Y, DT):
    pass


def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    pass


def compute_thresholds(X, f_index):
    # compute midpoints of X, return in a list
    midpoints = []

    unique = np.unique(X[:,f_index])
    np.sort(unique)

    for index in range(len(unique)-1):
        midpoints.append( (unique[index]+unique[index+1])/2 )

    return midpoints


def choose_best_tree(X, Y, feature_indices, current_depth, max_depth, is_binary=True):
    best_leq_indices = []
    best_gre_indices = []
    best_threshold = 0

    if same_labels(Y):
        return {"accuracy": 1.0, "output": Y[0][0]}
    elif same_features(X):
        label, count = highest_count(Y)
        return {"accuracy": count / len(Y), "output": highest_count(Y)[0]}
    elif current_depth == max_depth:
        label, count = highest_count(Y)
        return {"accuracy": count / len(Y), "output": highest_count(Y)[0]}

    best_acc = 0

    if X is None:
        return None

    for f_index in range(len(X[0])):
        if f_index in feature_indices:
            pass
        else:
            if is_binary:
                thresholds = [0.5]
            else:
                thresholds = compute_thresholds(X, f_index)

            for threshold in thresholds:
                (accuracy, leq_indices, gre_indices) = get_accuracy(X, Y, f_index, threshold)
                if (accuracy > best_acc):
                    feature_index = f_index
                    best_acc = accuracy
                    best_leq_indices = leq_indices
                    best_gre_indices = gre_indices
                    best_threshold = threshold

    feature_indices.append(feature_index)

    lower_tree = choose_best_tree(X[best_leq_indices], Y[best_leq_indices], feature_indices, current_depth + 1,
                                  max_depth, is_binary=is_binary)

    upper_tree = choose_best_tree(X[best_gre_indices], Y[best_gre_indices], feature_indices, current_depth + 1,
                                  max_depth, is_binary=is_binary)

    tree = {
        "accuracy": best_acc,
        "feature": feature_index,
        "threshold": best_threshold,
        (feature_index, 0): lower_tree,
        (feature_index, 1): upper_tree
    }

    return tree


def get_accuracy(samples, labels, feature, threshold):
    below = []
    above = []
    leq_indices = []
    gre_indices = []
    correct = {}

    for s_index, sample in enumerate(samples):
        if sample[feature] <= threshold:
            below.append(labels[s_index])
            leq_indices.append(s_index)
        else:
            above.append(labels[s_index])
            gre_indices.append(s_index)
    correct["leq"] = [below.count(0), below.count(1)]
    correct["gre"] = [above.count(0), above.count(1)]

    if len(leq_indices) == 0 or len(gre_indices) == 0:
        accuracy = 0
    else:
        accuracy = max(correct["leq"][0] + correct["gre"][1], correct["leq"][1] + correct["gre"][0]) / len(samples)

    return accuracy, leq_indices, gre_indices


def same_labels(vec):
    for val in vec:
        if not np.array_equal(val, vec[0]):
            return False
    return True


def same_features(X):
    for i in range(len(X)):
        for j in range(len(X)):
            if not np.array_equal(X[i], X[j]):
                return False
    return True


def highest_count(labels):
    unique, counts = np.unique(labels, return_counts=True)

    if not len(labels):
        return [None], 0

    max_count = 0

    for index, count in enumerate(counts):
        if count > max_count:
            max_count = count
            label = unique[index]

    return label, max_count


def make_prediction(sample, tree):
    if "output" in tree.keys():
        return tree["output"]
    else:
        feature = tree["feature"]
        if sample[feature] <= tree["threshold"]:
            return make_prediction(sample, tree[(feature, 0)])
        else:
            return make_prediction(sample, tree[(feature, 1)])





def main():
    tree1 = DT_train_binary(X_train_1, Y_train_1, -1)
    tree2 = DT_train_binary(X_train_2, Y_train_2, -1)
    print(DT_test_binary(X_test_1, Y_test_1, tree1))
    print(DT_test_binary(X_test_2, Y_test_2, tree2))
    tree3 = DT_train_real(X_real, Y_real, -1)


if __name__ == '__main__':
    main()
