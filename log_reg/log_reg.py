import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm


def read_data_set(path, add_intercept=False):
    # read data set at given path

    features = []
    classes = []

    with open(path, "r") as f:
        for line in f.readlines():
            sample = line.strip().split("\t")
            feat, label = sample[:-1], sample[-1]
            features.append([float(feature) for feature in feat])
            if add_intercept:
                features[-1].insert(0, 1)
            classes.append(int(label))

    return np.array(features), np.array(classes)


def split_data_set(x, y, train_percent):
    # split data set between train/test data

    assert 1 <= train_percent <= 100

    num_train = int(train_percent/100 * len(x))
    if num_train == 0:
        num_train = 1

    train_set = set(random.sample(range(len(x)), num_train))
    test_set = set(range(len(x))) - train_set

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for train_idx in train_set:
        train_x.append(x[train_idx])
        train_y.append(y[train_idx])

    for test_idx in test_set:
        test_x.append(x[test_idx])
        test_y.append(y[test_idx])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# computes sigmoid for a np array of values z


def sig(z): return 1/(1 + np.exp(-z))


# applies sigmoid for dot product between a vector of features x and a vector of weights w
# x.shape[1] == w.shape[0]
def apply_sig(x, w):
    return sig(np.dot(x, w))


def cost_function(x, y, w):
    """
    computes the logistic regression cost of a vector of features x, with their corresponding labels y
    and a vector of weights w
    """
    z = np.dot(x, w)
    return -sum(y*np.log(sig(z)) + (1 - y)*np.log(1 - sig(z)))/x.shape[0]


def log_reg_non_stoch(x, y, lr, max_iterations, print_iter_num, verbose=False):
    # computes the weights a vector of features x and their corresponding labels  y
    # using the non stochastic logistic regression algorithm

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    n, m = x.shape
    weights = np.random.rand(m)
    best_weights = weights
    best_cost = np.inf

    loss = []
    for iteration in tqdm(range(max_iterations)):
        diff = apply_sig(x, weights) - y
        weights = weights - lr/n * np.dot(x.T, diff)
        current_loss = cost_function(x, y, weights)
        if current_loss < best_cost:
            best_cost = current_loss
            best_weights = weights

        loss.append(current_loss)
        if iteration % print_iter_num == 0 and verbose is True:
            print(f"Iteration: {iteration}, has cost: {current_loss}")

    return best_weights, weights, loss


def log_reg_stoch(x, y, lr, max_iterations, print_iter_num, verbose=False):
    # computes the weights a vector of features x and their corresponding labels  y
    # using the non stochastic logistic regression algorithm

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    n, m = x.shape
    weights = np.random.rand(m)
    best_weights = weights
    best_cost = np.inf

    loss = []
    for iteration in tqdm(range(max_iterations)):
        for i in range(len(x)):
            diff = sig(np.dot(x[i], weights)) - y[i]
            weights = weights - lr/n * diff * x[i]

        current_loss = cost_function(x, y, weights)
        if current_loss < best_cost:
            best_cost = current_loss
            best_weights = weights

        loss.append(current_loss)
        if iteration % print_iter_num == 0 and verbose is True:
            print(f"Iteration: {iteration}, has cost: {current_loss}")

    return best_weights, weights, loss


def predict(x, w):
    # predicts the binary class of a vector of features x, using a vector of weights w
    return np.array([1 if pred > 0.5 else 0 for pred in apply_sig(x, w)])


def error_rate(x, y, w):
    # computes error rate for a vector of features x with the corresponding labels, using a vector of
    # weights w

    return sum(np.abs(predict(x, w) - y)) / len(y)


def normalize_data(x, n_type='linear'):
    # normalizes the input vector x
    # n_type can be 'z-score' or 'linear'

    if isinstance(x, list):
        x = np.arange(x)

    if n_type == 'linear':
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        for i in range(len(mins)):
            if mins[i] == maxs[i]:
                mins[i] = 0
        return (x - mins)/(maxs - mins)
    elif n_type == 'z-score':
        return (x - np.mean(x, axis=0))/np.std(x, axis=0)
    else:
        return None


def plot_2d_data(x, y):
    # plot 2d data

    x_0 = [x[i][1] for i in range(len(x)) if y[i] == 0]
    y_0 = [x[i][2] for i in range(len(x)) if y[i] == 0]

    x_1 = [x[i][1] for i in range(len(x)) if y[i] == 1]
    y_1 = [x[i][2] for i in range(len(x)) if y[i] == 1]

    plt.plot(x_0, y_0, "o", label='0 class')
    plt.plot(x_1, y_1, "s", label='1 class')
    plt.legend()
    plt.show()


def plot_loss(loss):
    plt.plot(range(len(loss)), loss)
    plt.show()


def plot_2d_decision_boundary(x, y, w):
    x_0 = [x[i][1] for i in range(len(x)) if y[i] == 0]
    y_0 = [x[i][2] for i in range(len(x)) if y[i] == 0]

    x_1 = [x[i][1] for i in range(len(x)) if y[i] == 1]
    y_1 = [x[i][2] for i in range(len(x)) if y[i] == 1]

    x_line = np.arange(0, 1, 0.01)
    y_line = (-w[0] - w[1]*x_line)/w[2]

    plt.plot(x_0, y_0, "o", label='0 class')
    plt.plot(x_1, y_1, "s", label='1 class')
    plt.plot(x_line, y_line, label='Decision line')
    plt.legend()
    plt.show()
