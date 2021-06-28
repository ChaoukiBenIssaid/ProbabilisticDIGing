import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from graph_utils import *
from utils import * 

def sim_linear_regression(X, y, n, iter_max, learning_rate, algo = "dgd", proba = prob_gatien, proba_param = 1, standardize = False, seed=None):
    """Simulate a linear regression between n agents
    
    need to do the docstring 
    
    algo : "dgd", "dig" or "prob-dig"
    """
    rng = np.random.default_rng(seed)
    ##### DATASET PROCESSING
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
    ######
    ###### ALGORITHM INITIALIZATION
    thetas = [rng.standard_normal(nb_features) for _ in range(n)] 
    deltas = [lr_grad(thetas[i], X_train[i], y_train[i]) for i in range(n)]

    losses = list() 
    communication_costs = [0]
    ######
    ###### ALGORITHM
    for k in tqdm(range(iter_max)):
        ### Dynamic graph
        A = generate_random_adjacency_matrix(n, seed)
        W = metropolis_weights(A)
        degs = degrees(A)
        
        ### Linear regression initialization
        new_thetas = thetas.copy() # to change all the thetas/delta at once
        new_deltas = deltas.copy()
        c_k = 0 # communication costs at iteration k 
        ### Linear regression
        for i in range(n):
            ## theta update
            grad = lr_grad(thetas[i], X_train[i], y_train[i]) # local gradient of agent i at iteration k 
            Wtheta = [W[i,j]*thetas[j] for j in range(n)] 
            sum_Wtheta = np.sum(Wtheta, axis = 0) # first part of theta update for agent i 
            new_thetas[i] = sum_Wtheta - learning_rate * deltas[i] # update
            
            ##delta update
            # algorithm choice
            if algo in ["dgd", "dig"] : 
                use_dgd = algo == "dgd" # = 1 if we use DGD *
            elif algo == "prob-dig": 
                use_dgd = rng.random() < proba(k, proba_param) #1 if we use DGD and 0 if we use DIG 

            # update
            new_grad = lr_grad(new_thetas[i], X_train[i], y_train[i]) # local gradient of agent i at iteration k + 1    
            if use_dgd : 
                new_deltas[i] = new_grad
                c_k += nb_features*int(degs[i]) 
            else:
                Wdelta = [W[i,j]*deltas[j] for j in range(n)]
                sum_Wdelta = np.sum(Wdelta, axis = 0)
                new_deltas[i] = sum_Wdelta + new_grad - grad
                c_k += 2*nb_features*int(degs[i])
        ### thetas and deltas update
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