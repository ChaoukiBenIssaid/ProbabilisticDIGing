import numpy as np
import networkx as nx

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
    perm = rng.permutation(y.size)[y.size % n:]
    X_split = np.stack(np.array_split(X[perm], n), axis=1)  #np.stack to keep as a np array
    y_split = np.stack(np.array_split(y[perm], n), axis=1)
    return X_split, y_split

def metropolis_weights(A):
    """Apply the Metropolis weights algorithm to an adjacency matrix"""
    deg = degrees(A)
    W_without_diag = A / (1 + np.maximum(deg, deg.T)) #generate the weights for i != j
    diag = np.diag(np.ones(deg.size) - np.sum(W_without_diag, axis=1)) #generate the weights for i=j
    return W_without_diag + diag 

