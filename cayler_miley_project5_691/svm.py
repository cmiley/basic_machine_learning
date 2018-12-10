import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def svm_train_brute(data):
    pos_data = [point for point in data if point[-1] > 0]
    neg_data = [point for point in data if point[-1] <= 0]

    # find the closest points between the two classes
    pos, neg = find_closest_points(pos_data, neg_data)

    w = (pos[:-1] - neg[:-1])
    margin = np.linalg.norm(w)/2
    w = w/np.linalg.norm(w) * (1/margin)
    b = 1 - w.dot(pos[:-1])

    for negative in neg_data:
        if not np.all([neg, negative]):
            w_dir = get_w_dir(pos, neg, negative)
            print(w_dir)

    support_vectors = compute_support_vecs(data, w, b)

    return np.array([w, b, support_vectors])


def distance_point_to_hyperplane(pt, w, b):
    dist = np.absolute(np.dot(pt, w) + b)/np.linalg.norm(w)

    return dist


def compute_margin(data, w, b, cls=1):
    pos_margin = neg_margin = 1e20

    for point in data:
        distance = distance_point_to_hyperplane(point[:-1], w, b)
        if point[-1] == cls:
            pos_margin = np.minimum(pos_margin, distance)
        else:
            neg_margin = np.minimum(neg_margin, distance)

    margin = pos_margin + neg_margin

    return margin/2


def find_closest_points(positive_class, negative_class):
    min_dist = 1e20
    pos_closest = []
    neg_closest = []

    for pos_point in positive_class:
        for neg_point in negative_class:
            dist = np.linalg.norm(pos_point[:-1] - neg_point[:-1])
            if dist < min_dist:
                pos_closest = [pos_point]
                neg_closest = [neg_point]
                min_dist = dist
            elif dist == min_dist:
                pos_closest.append(pos_point)
                neg_closest.append(neg_point)

    return np.mean(pos_closest, axis=0), np.mean(neg_closest, axis=0)


def plot_decision_boundary_binary(data, w, b, save=False, title=None):
    plt.figure()
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'g+')
        else:
            plt.plot(item[0], item[1], 'ro')
    m = max(data.max(), abs(data.min())) + 1

    margin = compute_margin(data, w, b)

    xx = np.linspace(-m, m)
    # boundaries are not vertical
    if w[1] != 0:
        yy = (-w[0] * xx - b) / w[1]
        # boundaries are horizontal
        if w[0] == 0:
            yy_right = yy + margin
            yy_left = yy - margin
        # boundaries are diagonal
        else:
            yy_right = yy + margin * np.sqrt(2)
            yy_left = yy - margin * np.sqrt(2)

        plt.plot(xx, yy_right, ':b')
        plt.plot(xx, yy_left, ':b')
        plt.plot(xx, yy, '-k')
    # decision boundaries are vertical
    else:
        plt.axvline(x=b, color='k', linestyle='-')
        plt.axvline(x=b+margin, color='b', linestyle=':')
        plt.axvline(x=b-margin, color='b', linestyle=':')

    plt.axis([-m, m, -m, m])
    if save:
        plt.savefig('{}.png'.format(title))
    else:
        plt.show()


def compute_support_vecs(data, w, b):
    support = []
    for point in data:
        if round(w.dot(point[:-1]) + b) == point[-1]:
            support.append(point)

    return support


def svm_test_brute(w, b, x):
    if w.dot(x) + b > 0:
        y_hat = 1
    else:
        y_hat = -1
    return y_hat


def get_w_dir(pos_close, neg_close, pt):
    pos_pt = pos_close[:-1]
    neg_pt = neg_close[:-1]
    if pt[-1] == 1:
        same = pos_pt - pt[:-1]
        diff = neg_pt - pt[:-1]
    else:
        same = neg_pt - pt[:-1]
        diff = pos_pt - pt[:-1]

    w_dir = (diff.dot(same)/np.linalg.norm(diff)**2) * diff

    return w_dir


def svm_train_multiclass(training_data):
    num_classes = int(training_data[:, -1].max())
    W_mat = np.empty([num_classes, training_data[0, :-1].size])
    B_vec = np.empty(num_classes)

    for classi in range(num_classes):
        labels = training_data[:, -1].copy()
        for idx, label in enumerate(labels):
            if label == classi + 1:
                labels[idx] = 1
            else:
                labels[idx] = -1
        one_v_all_data = np.concatenate((training_data[:, :-1], labels.reshape(-1, 1)), axis=1)

        w, b, S = svm_train_brute(one_v_all_data)
        W_mat[classi] = w
        B_vec[classi] = b

    return W_mat, B_vec


def svm_test_multiclass(w, b, x):
    pass


def plot_decision_boundary_multi(data, W, B, save=False, title=None):
    plt.figure()
    colors = ['b', 'r', 'g', 'm']
    shapes = ['+', 'o', '*', '.']

    for item in data:
        plt.plot(item[0], item[1], colors[int(item[2]) - 1] + shapes[int(item[2]) - 1])
    m = max(data.max(), abs(data.min())) + 1
    plt.axis([-m, m, -m, m])
    xx = np.linspace(-m, m)

    for idx, w in enumerate(W):

        margin = compute_margin(data, w, B[idx], idx+1)
        # print(margin)
        # boundaries are not vertical
        if w[1] != 0:
            yy = (-w[0] * xx - B[idx]) / w[1]
            if w[0] == 0:
                yy_right = yy + margin
                yy_left = yy - margin
            # boundaries are diagonal
            else:
                yy_right = yy + margin * np.sqrt(2)
                yy_left = yy - margin * np.sqrt(2)

            # boundaries are horizontal
            plt.plot(xx, yy, color=colors[idx], linestyle=':')
            plt.plot(xx, yy_right, ':b')
            plt.plot(xx, yy_left, ':b')
        # decision boundaries are vertical
        else:
            plt.axvline(x=B[idx], color=colors[idx], linestyle=':')
            plt.axvline(x=B[idx] + margin, color='b', linestyle=':')
            plt.axvline(x=B[idx] - margin, color='b', linestyle=':')

    if save:
        plt.savefig('multi_{}.png'.format(title))
    else:
        plt.show()
