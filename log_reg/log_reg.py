import numpy as np
import numpy.random as rand
from tqdm import tqdm


def sig(z): return 1/(1 + np.exp(-z))


def apply_sig(x, w):
    return sig(np.dot(x, w))


def cost_function(x, y, w):
    z = np.dot(x, w)
    return -sum(y*np.log(sig(z)) + (1 - y)*np.log(1 - sig(z)))/x.shape[0]


def log_reg_non_stoch(x, y, lr, max_iterations):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    n, m = x.shape
    weights = rand(m)
    loss = []
    for iteration in tqdm(range(max_iterations)):
        diff = y - apply_sig(x, weights)
        weights = weights - lr/n * np.dot(x.T, diff)
        loss.append(cost_function(x, y, weights))

    return weights, loss


def predict(x, w):
    return [1 if pred > 0.5 else 0 for pred in apply_sig(x, w)]
