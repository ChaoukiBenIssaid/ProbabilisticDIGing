import numpy as np
import networkx as nx
from graph_utils import *
from tqdm import tqdm


def MSE(X, y, theta):
    """Compute the MSE"""
    step1 = 1 / X.shape[0] * np.einsum("ijk,jk->ij", X, theta) - y
    return 1 / 2 * np.linalg.norm(step1, axis=0)**2


def grad_MSE(X, y, theta):
    """Compute the gradient of the MSE at a point"""
    step1 = np.einsum("ijk,jk->ij", X, theta) - y
    step2 = np.einsum("ij,ijk->jk", step1, X)
    return 1 / X.shape[0] * step2

def sim_linear_regression(X, y, n, iter_max, learning_rate, seed=None):
    """Simulate a linear regression between n agents"""
    A = generate_random_adjacency_matrix(n)
    W = metropolis_weights(A)
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1]
    theta = rng.standard_normal(nb_features * n).reshape(n, nb_features)
    X_split, y_split = random_split(X, y, n, seed)

    gradients_list = list()

    for _ in range(iter_max):
        grad = grad_MSE(X_split, y_split, theta)
        theta = W @ theta - learning_rate * grad

        gradients_list.append(grad)
    return gradients_list
