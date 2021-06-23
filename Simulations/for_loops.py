import numpy as np
import networkx as nx 
from graph_utils import *
from utils import *
from sklearn.preprocessing import StandardScaler

def random_split(X, y, n, seed=None):
    """Equally split data between n agents"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y.size)
    X_split = np.array_split(X[perm], n)  #np.stack to keep as a np array
    y_split = np.array_split(y[perm], n)
    return X_split, y_split

def sim_linear_regression(X, y, n, iter_max, learning_rate, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    A = generate_random_adjacency_matrix(n)
    W = metropolis_weights(A)
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    X_split, y_split = random_split(X, y, n, seed = seed)
    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 

    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_split[i] = scalers[i].fit_transform(X_split[i]) #scale each agent's dataset independently

    X_split = [np.append(X_split[i], np.ones((X_split[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones

    thetas_tracker = [thetas]
    for _ in range(iter_max):
        new_thetas = thetas.copy()
        for i in range(n):
            grad = 1/X_split[i].shape[0] * lr_grad(thetas[i], X_split[i], y_split[i])
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            new_thetas[i] = sum_Wtheta - learning_rate * grad
        thetas = new_thetas
        thetas_tracker.append(thetas)
    return thetas_tracker

