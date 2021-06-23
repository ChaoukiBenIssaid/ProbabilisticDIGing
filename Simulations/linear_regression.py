import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_boston

def generate_random_adjacency_matrix(n):
    """Generate the adjacency matrix of a random connected graph"""
    while True:
        g = nx.generators.random_graphs.binomial_graph(n, 0.4)
        if nx.algorithms.components.is_connected(g):
            return nx.linalg.graphmatrix.adjacency_matrix(g)
        
def degrees(A):
    """Return the degrees of each node of a graph from its adjacency matrix"""
    return np.sum(A, axis=0).reshape(A.shape[0], 1)

def adjacency_matrix(W):
    """Return the adjacency matrix from a weights matrix"""
    return (W > 0) * (1 - np.diag(np.ones(W.shape[0])))  #We want a 1 for each positive edge but the diagonal.

def neighbors(W, i):
    """Return the indices of node i's neighbors from a weights matrix"""
    adj = adjacency_matrix(W)
    return np.where(adj[i] == 1)

def random_split(X, y, n, seed=None):
    """Equally split data between n agents"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y.size)
    X_split = np.array_split(X[perm], n)  #np.stack to keep as a np array
    y_split = np.array_split(y[perm], n)
    return X_split, y_split

def metropolis_weights(Adj):
    N = np.shape(Adj)[0]
    degree = degrees(Adj)
    W = np.zeros([N, N])
    for i in range(N):
        N_i = np.nonzero(Adj[i, :])[1]  # Fixed Neighbors
        for j in N_i:
            W[i, j] = 1/(1+np.max([degree[i], degree[j]]))
        W[i, i] = 1 - np.sum(W[i, :])
    return W

def lr_grad(W, X, y):
    return X.T.dot(X.dot(W) - y)

def sim_linear_regression(X, y, n, iter_max, learning_rate, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    A = generate_random_adjacency_matrix(n)
    W = metropolis_weights(A)
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i]) for i in range(n)]
    X_train, X_test, y_train, y_test = [[tts_agent[i] for tts_agent in train_test_datasets] for i in range(4)]




    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_train[i] = scalers[i].fit_transform(X_train[i]) #scale each agent's dataset independently
            X_test[i] = scalers[i].transform(X_test[i]) 

    X_train = [np.append(X_train[i], np.ones((X_train[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones
    X_test = [np.append(X_test[i], np.ones((X_test[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones



    thetas_tracker = [thetas]
    for _ in range(iter_max):
        new_thetas = thetas.copy()
        for i in range(n):
            grad = lr_grad(thetas[i], X_train[i], y_train[i])
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            new_thetas[i] = sum_Wtheta - learning_rate * grad
        thetas = new_thetas
        thetas_tracker.append(thetas)
    return thetas_tracker