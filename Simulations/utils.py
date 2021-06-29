import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import numpy as np
from random import random
from sklearn.model_selection import train_test_split 


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
    ywTx = y * (X @ W)
    temp = 1. / (1. + np.exp(ywTx))
    grad = -(X.T @ (y * temp)) + lmbd * W
    return grad

# linear regression loss function
def lr_loss(W, X, y):
    diff = y - X.dot(W)
    return 0.5/X.shape[0] * diff.dot(diff)

# linear regression gradient function
def lr_grad(W, X, y):
    return 1/X.shape[0] * X.T.dot(X.dot(W) - y)

def accuracy(W, X, y, activation_function):
    pred = activation_function(X@W)
    return np.sum(pred == y)/len(y) 
    
def prob_gatien(k, a) : 
    return a / (a + k) 

def prob_chaouki(k, T) : 
    return np.exp(-k/T)

def random_split(X, y, n, seed=None):
    """Equally split data between n agents"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y.size)
    X_split = np.array_split(X[perm], n)  #np.stack to keep as a np array
    y_split = np.array_split(y[perm], n)
    return X_split, y_split

def split_to_train_test(X, y, n, seed = None):
    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i], random_state = seed) for i in range(n)]
    return train_test_datasets
# more functions if needed

# Draw the graph
# nx.draw(G, with_labels = True)

# Get the adjacency matrix
# adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float)

# get the neighbours for the ith worker
# neighbors = np.where(A[i,:]==1)[1]
