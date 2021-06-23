%matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import numpy as np
from random import random

#number of workers
n = 10

def generate_graph(n, p):
    V = set([v for v in range(n)])
    E = set()
    for combination in combinations(V, 2):
        a = random()
        if a < p:
            E.add(combination)

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g

def generate_connected_graph(n,p):
    p = 0.4 
    G = ER(n, p)
    while not nx.is_connected(G):
        G = ER(n, p)
    
    return G

# weighted or mixing matrix using Metropolis-Hastings method
def mixing_matrix(Adj):   
    
    N = np.shape(Adj)[0]
    degree = np.sum(Adj, axis=0)
    W = np.zeros([N, N])
    for i in range(N):
        N_i = np.nonzero(Adj[i, :])[0]  # Fixed Neighbors
        for j in N_i:
            W[i, j] = 1/(1+np.max([degree[i], degree[j]]))
        W[i, i] = 1 - np.sum(W[i, :])
    return W

def logistic_loss(W, X, y, lmbd):
    pr = y*X.dot(W)
    l2 = 0.5 * np.dot(W, W)
    return np.log1p(np.exp(-pr)).sum() + lmbd * l2

# gradient function for logistic regression case
def logistic_grad(W, X, y, lmbd):
    ywTx = y * (X @ w)
    temp = 1. / (1. + np.exp(ywTx))
    grad = -(X.T @ (y * temp)) + lmbd * w
    return grad

# linear regression loss function
def lr_loss(W, X, y):
    diff = y - X.dot(W)
    return 0.5 * diff.dot(diff)

# linear regression gradient function
def lr_grad(W, X, y):
    return X.T.dot(X.dot(W) - y)


# more functions if needed

# Draw the graph
# nx.draw(G, with_labels = True)

# Get the adjacency matrix
# adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float)

# get the neighbours for the ith worker
# neighbors = np.where(A[i,:]==1)[1]
