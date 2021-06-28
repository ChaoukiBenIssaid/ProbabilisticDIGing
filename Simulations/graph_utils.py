import numpy as np
import networkx as nx

def generate_random_adjacency_matrix(n, seed):
    """Generate the adjacency matrix of a random connected graph"""
    while True:
        g = nx.generators.random_graphs.binomial_graph(n, 0.4, seed = seed)
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

