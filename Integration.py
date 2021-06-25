# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:54:37 2021

@author: Gabriel Alejandro Morales Ruiz
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from L1eSVR import L1e_SVR
import numpy as np
import matplotlib.pyplot as plt
from GaussianOptimization import GaussianOptimization
from sklearn import preprocessing
from sklearn.utils import check_random_state

# Global variables
e_ = 0.2
sigma_ = 2
results = []

def plot(X, y, model):
    """
    Simple function to plot the model and objective function plots (scatter)

    Parameters
    ----------
    X : np.list
        Objective function samples' input.
    y : np.list
        Objective function samples' output.
    model : TYPE
        Gaussian Optimization model.

    Returns
    -------
    None.

    """
    plt.scatter(X, y) # Scatter plot of real pairs
    # Sample surrogate function
    Xsamples  = np.linspace(min(X), max(X), 200).reshape(200, 1)
    ysamples, _ = model.surrogate(Xsamples)
    plt.plot(Xsamples, ysamples) # Continuous plot of surrogate function
    plt.show()

#%% Database
rng = check_random_state(0) # Seed for reproducibility

boston = load_boston()

# Shuffle
perm = rng.permutation(boston.target.size) 
boston.data = boston.data[perm]
boston.target = boston.target[perm]

# Assign
y = boston.target
y=np.reshape(y,(len(boston.target),1))
X = boston.data

# Standarize
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
scaler = preprocessing.StandardScaler().fit(y)
y_scaled = scaler.transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y_scaled,
                                                    test_size=0.6,
                                                    random_state = 6)
n_train = X_train.shape[0]

#%% SVR without hyperparameter optimization

svr_base = L1e_SVR(k_type="rbf", c=1, e=e_)
svr_base.fit(X_train, y_train)
r2_base = svr_base.score(X_test, y_test)
print("Base model's score: " + str(r2_base))


#%% Optimization functions

# R^2 score vs C normalization hyperparameter.
def objective1(c, X=X_train, y=y_train, X_test=X_test, y_test=y_test) :
    global e_
    global sigma_
    global results
    svr = L1e_SVR(k_type="rbf", c=c, e=e_)
    svr.fit(X, y)
    r2 = svr.score(X_test, y_test)
    results.append([c, e_, sigma_, r2])
    return r2

def sample_objective1() :
    C = np.linspace(1, 20, 5)
    r2 = []
    for c in C :
        #print("Using c=" + str(c))
        r2.append(objective1(c))
    r2 = np.asarray(r2)
    C_mat = C.reshape(len(C), 1)
    r2_mat = r2.reshape(len(r2), 1)
    return C_mat, r2_mat

obj1_best_c = None

# R^2 score vs epsilon (margin) hyperparameter.
def objective2(e, X=X_train, y=y_train, X_test=X_test, y_test=y_test) :
    global e_
    global sigma_
    global obj1_best_c
    e_ = e
    C_mat, r2_mat = sample_objective1()
    model_c = GaussianOptimization(objective1,
                                   C_mat,
                                   r2_mat,
                                   minimize = False,
                                   l = 3)
    model_c.optimize(iters=10)
    ix = np.argmax(model_c.y)
    obj1_best_c = model_c.X[ix]
    svr = L1e_SVR(k_type="rbf", c = obj1_best_c, e=e_, sigma=sigma_)
    svr.fit(X, y)
    r2 = svr.score(X_test, y_test)
    return r2

def sample_objective2() :
    E = np.linspace(0.01, 2, 5)
    r2 = []
    for e in E :
        #print("Using e=" + str(e))
        r2.append(objective2(e))
    r2 = np.asarray(r2) 
    E_mat = E.reshape(len(E), 1)
    r2_mat = r2.reshape(len(r2), 1)
    return E_mat, r2_mat

# R^2 score vs kernel (sigma) hyperparameter.
def objective3(sigma, X=X_train, y=y_train, X_test=X_test, y_test=y_test) :
    global obj1_best_c
    global sigma_
    sigma_ = sigma
    E_mat, r2_mat = sample_objective2()
    model_e = GaussianOptimization(objective2,
                                   E_mat,
                                   r2_mat,
                                   minimize = False,
                                   l = 0.5)
    model_e.optimize(iters=7)
    ix = np.argmax(model_e.y)
    svr = L1e_SVR(k_type="rbf", c=obj1_best_c, e=model_e.X[ix], sigma=sigma_)
    svr.fit(X, y)
    r2 = svr.score(X_test, y_test)
    return r2

def sample_objective3() :
    S = np.linspace(0.1, 5, 5)
    r2 = []
    for s in S :
        print("Using s=" + str(s))
        r2.append(objective3(s))
    r2 = np.asarray(r2) 
    S_mat = S.reshape(len(S), 1)
    r2_mat = r2.reshape(len(r2), 1)
    return S_mat, r2_mat

#%% Optimization process
S_mat, r2_mat = sample_objective3()
model = GaussianOptimization(objective3,
                             S_mat,
                             r2_mat,
                             minimize = False,
                             l=1)
model.optimize(iters=10)

#%% Results extraction
res_mat = np.matrix(results)
c_list = np.array(res_mat[:, 0]).ravel()
e_list = np.array(res_mat[:, 1]).ravel()
s_list = np.array(res_mat[:, 2]).ravel()
r2_list = np.array(res_mat[:, 3]).ravel()

ix = np.argmax(r2_list)
print("Best result:")
print("  c = " +  str(c_list[ix]))
print("  e = " +  str(e_list[ix]))
print("  sigma = " +  str(s_list[ix]))
print("  r^2 = " +  str(r2_list[ix]))


#%% OLS Regression
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import r2_score
data = pd.DataFrame(np.hstack((X_train, y_train)))
X_ols = pd.DataFrame(X_train)
y_ols = pd.DataFrame(y_train)
mlr = sm.OLS(y_ols, X_ols).fit()    

X_ols_test = pd.DataFrame(X_test)
y_ols_test = pd.DataFrame(y_test)

sm.add_constant(X_ols_test)
y_ols_pred = mlr.predict(X_ols_test)
r2_ols = r2_score(y_test, y_ols_pred)
print("Ordinary Least Squares Regression")
print(" r2: " + str(r2_ols))
