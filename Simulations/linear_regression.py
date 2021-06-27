import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
#from sklearn.datasets import load_boston

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


def sim_linear_regression(X, y, n, iter_max, learning_rate, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i], random_state = seed) for i in range(n)]
    X_train, X_test, y_train, y_test = [[tts_agent[i] for tts_agent in train_test_datasets] for i in range(4)]




    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_train[i] = scalers[i].fit_transform(X_train[i]) #scale each agent's dataset independently
            X_test[i] = scalers[i].transform(X_test[i]) 

    X_train = [np.append(X_train[i], np.ones((X_train[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones
    X_test = [np.append(X_test[i], np.ones((X_test[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones



    losses = list() 
    communication_costs = [0]
    for _ in range(iter_max):
        #Dynamic graph
        A = generate_random_adjacency_matrix(n, seed)
        W = metropolis_weights(A)
        #
        new_thetas = thetas.copy()
        for i in range(n):
            grad = lr_grad(thetas[i], X_train[i], y_train[i])
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            new_thetas[i] = sum_Wtheta - learning_rate * grad
        thetas = new_thetas
        loss = np.mean([lr_loss(thetas[i], X_test[i], y_test[i]) for i in range(n)])
        losses.append(loss)
    
        #communication costs part 
        nb_comms = np.sum(degrees(A))
        communication_costs.append(nb_features * nb_comms + communication_costs[-1])


    results = dict()
    results["losses"] = losses
    results["comm_costs"] = communication_costs[1:]
    return results 

def sim_logistic_regression(X, y, n, iter_max, learning_rate, lmbd = 0, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    A = generate_random_adjacency_matrix(n)
    W = metropolis_weights(A)
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i], random_state = seed) for i in range(n)]
    X_train, X_test, y_train, y_test = [[tts_agent[i] for tts_agent in train_test_datasets] for i in range(4)]




    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_train[i] = scalers[i].fit_transform(X_train[i]) #scale each agent's dataset independently
            X_test[i] = scalers[i].transform(X_test[i]) 

    X_train = [np.append(X_train[i], np.ones((X_train[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones
    X_test = [np.append(X_test[i], np.ones((X_test[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones



    losses = list() 
    for _ in range(iter_max):
        new_thetas = thetas.copy()
        for i in range(n):
            grad = logistic_grad(thetas[i], X_train[i], y_train[i], lmbd)
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            new_thetas[i] = sum_Wtheta - learning_rate * grad
        thetas = new_thetas
        loss = np.mean([logistic_loss(thetas[i], X_test[i], y_test[i], lmbd) for i in range(n)])
        losses.append(loss)
    return losses



def DIGing_linear_reg(X, y, n, iter_max, learning_rate, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i], random_state = seed) for i in range(n)]
    X_train, X_test, y_train, y_test = [[tts_agent[i] for tts_agent in train_test_datasets] for i in range(4)]
    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_train[i] = scalers[i].fit_transform(X_train[i]) #scale each agent's dataset independently
            X_test[i] = scalers[i].transform(X_test[i]) 

    X_train = [np.append(X_train[i], np.ones((X_train[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones
    X_test = [np.append(X_test[i], np.ones((X_test[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones

    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    deltas = [lr_grad(thetas[i], X_train[i], y_train[i]) for i in range(n)]


    losses = list() 
    communication_costs = [0]
    for _ in range(iter_max):
        #Dynamic graph
        A = generate_random_adjacency_matrix(n, seed)
        W = metropolis_weights(A)
        #
        new_thetas = thetas.copy()
        new_deltas = deltas.copy()
        
        for i in range(n):
            #theta update
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            new_thetas[i] = sum_Wtheta - learning_rate * new_deltas[i]

            #delta update
            Wdelta = [W[i,j]*deltas[j] for j in range(n)]
            sum_Wdelta = np.sum(Wdelta, axis = 0)
            grad = lr_grad(thetas[i], X_train[i], y_train[i])
            new_grad = lr_grad(new_thetas[i], X_train[i], y_train[i])
            new_deltas[i] = sum_Wdelta + new_grad - grad
        thetas = new_thetas
        deltas = new_deltas
        loss = np.mean([lr_loss(thetas[i], X_test[i], y_test[i]) for i in range(n)])
        losses.append(loss)
    
        #communication costs part 
        nb_comms = np.sum(degrees(A))
        communication_costs.append(2*nb_features * nb_comms + communication_costs[-1])


    results = dict()
    results["losses"] = losses
    results["comm_costs"] = communication_costs[1:]
    return results

def prob_DIGing_linear_reg(X, y, n, iter_max, learning_rate, proba = prob_gatien, proba_param = 1, standardize = False, seed=None):
    """Simulate a linear regression between n agents"""
    rng = np.random.default_rng(seed)
    nb_features = X.shape[-1] +1  #+1 for bias


    X_split, y_split = random_split(X, y, n, seed = seed)
    train_test_datasets = [train_test_split(X_split[i], y_split[i], random_state = seed) for i in range(n)]
    X_train, X_test, y_train, y_test = [[tts_agent[i] for tts_agent in train_test_datasets] for i in range(4)]
    if standardize:
        scalers = [StandardScaler() for _ in range(n)]
        for i in range(n) : 
            X_train[i] = scalers[i].fit_transform(X_train[i]) #scale each agent's dataset independently
            X_test[i] = scalers[i].transform(X_test[i]) 

    X_train = [np.append(X_train[i], np.ones((X_train[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones
    X_test = [np.append(X_test[i], np.ones((X_test[i].shape[0],1)), axis = 1) for i in range(n)] #add a column of ones

    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    deltas = [lr_grad(thetas[i], X_train[i], y_train[i]) for i in range(n)]


    losses = list() 
    communication_costs = [0]
    for k in range(iter_max):
        #Dynamic graph
        A = generate_random_adjacency_matrix(n, seed)
        W = metropolis_weights(A)
        degs = degrees(A)
        #
        new_thetas = thetas.copy()
        new_deltas = deltas.copy()
        
        c_k = 0
        for i in range(n):
            #theta update
            grad = lr_grad(thetas[i], X_train[i], y_train[i])
            Wtheta = [W[i,j]*thetas[j] for j in range(n)]
            sum_Wtheta = np.sum(Wtheta, axis = 0)
            bernoulli = rng.random() > proba(k, proba_param) #1 if we use DIG and 0 if we use DGD 
            if bernoulli:
                new_thetas[i] = sum_Wtheta - learning_rate * new_deltas[i]
                c_k += 2*nb_features*int(degs[i])
            else:
                new_thetas[i] = sum_Wtheta - learning_rate * grad
                c_k += nb_features*int(degs[i])
            #delta update
            Wdelta = [W[i,j]*deltas[j] for j in range(n)]
            sum_Wdelta = np.sum(Wdelta, axis = 0)
            new_grad = lr_grad(new_thetas[i], X_train[i], y_train[i])
            new_deltas[i] = sum_Wdelta + new_grad - grad
        thetas = new_thetas
        deltas = new_deltas
        loss = np.mean([lr_loss(thetas[i], X_test[i], y_test[i]) for i in range(n)])
        losses.append(loss)
    
        #communication costs part 
        communication_costs.append(c_k + communication_costs[-1])


    results = dict()
    results["losses"] = losses
    results["comm_costs"] = communication_costs[1:]
    return results