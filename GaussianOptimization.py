# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:56:03 2021

@author: Gabriel A. Morales Ruiz
"""

import numpy as np
from scipy.stats import norm

class GaussianProcessRegression :
    def __init__(self, l) :
        """
        Constructor.

        Parameters
        ----------
        l : float
            Parameter used on the Gaussian Process.

        Returns
        -------
        None.

        """
        self.l = l
        self.k = None
        self.X = None
        self.y = None
        
    def K(self, xk, xl) :
        """
        Gaussian process' kernel function

        Parameters
        ----------
        xk : np.matrix
            Feature vector.
        xl : np.matrix
            Feature vector.

        Returns
        -------
        np.matrix
            Output kernel.

        """
        v1k = np.ones((1, xk.shape[0]))
        v1l = np.ones((1, xl.shape[0]))
        Xk = xk*v1l;
        Xl = (xl*v1k).T
        return np.exp(-np.square(Xk - Xl)/(2*self.l))
    
    def fit(self, X, y) :
        """
        Stores inputs and outputs, and calculates the kernel.

        Parameters
        ----------
        X : np.matrix
            Input features.
        y : TYPE
            Expected output.

        Returns
        -------
        None.

        """
        self.X = X
        self.y = y
        self.k = self.K(X, X)
        
    def predict(self, X_test, return_std = False) :
        """
        Uses the fitted information to predict an output based on new features.

        Parameters
        ----------
        X_test : np.matrix
            New features to used on prediction
        return_std : boolean, optional
            Use if you want this function to calculate and return
            the prediction's standard deviation. The default is False.

        Returns
        -------
        mu : float
            Mean value of prediction.
        s : float
            Standard deviation of prediction.

        """
        # Mean
        L = np.linalg.cholesky(self.k + 1e-6*np.eye(self.k.shape[0]))
        Lk = np.linalg.solve(L, self.K(self.X, X_test))
        mu = np.dot(Lk.T, np.linalg.solve(L, self.y))
        
        # Variance        
        if return_std : 
            kss = self.K(X_test, X_test)
            s2 = np.diag(kss) - np.sum(np.square(Lk), axis=0)
            s = np.sqrt(s2)
            return mu, s
        else : return mu
        
class GaussianOptimization :
    def __init__(self, objective, X, y, l=1, minimize = True, pool=100) :
        """
        Constructor

        Parameters
        ----------
        objective : function pointer
            Pointer to .
        X : np.matrix
            Initial samples' input.
        y : np.matrix
            Initial samples' output.
        l : float, optional
            Gaussian process' hyperparameter. The default is 1.
        minimize : boolean, optional
            If false, then the optimization will maximize. The default is True.
        pool : int, optional
            Amount of points to use when calculating the next optimal point.
            The default is 100.

        Returns
        -------
        None.

        """
        self.gpr = GaussianProcessRegression(l)
        self.X = X
        self.y = y
        self.objective = objective
        self.minimize = minimize
        self.pool = pool
        self.gpr.fit(X, y)
        
    def surrogate(self, X) :
        """
        Uses the gaussian regression's function to simulate the objective
        function's output.

        Parameters
        ----------
        X : np.matrix
            Input to gaussian regression.

        Returns
        -------
        np.matrix
            Predicted value.

        """
        return self.gpr.predict(X, True)
    
    def acquire(self) :
        """
        Creates samples and runs them through the gaussian regression
        prediction. Then, this function uses current data to calculate whether
        one of those points has a higher chance of yielding a better result.

        Returns
        -------
        float
            Optimal point to use on objective function.

        """
        X_new = np.random.uniform(min(self.X), max(self.X), self.pool)
        # X_new = np.random.random(self.pool)
        X_new = X_new.reshape(self.pool, 1)
        
        
        y_pred, _ = self.surrogate(self.X)
        if self.minimize : best = min(y_pred)
        else :             best = max(y_pred)
        
        mu, std = self.surrogate(X_new)
        mu = mu[:, 0]
        
        probs = norm.cdf((mu - best)/(std + 1e-9))
        
        ix = np.argmax(probs)
        return X_new[ix, 0]
    
    def optimize(self, iters = 100) :
        """
        Runs the acquire function and evalutes on objective function with the
        purpose to find the max/min value.

        Parameters
        ----------
        iters : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
        for i in range(iters) :
            print("  Generation " + str(i))
            x = self.acquire()
            y = self.objective(x)
            
            mu, _ = self.surrogate(np.matrix(x))
            self.X = np.vstack((self.X, [[x]]))
            self.y = np.vstack((self.y, [[y]]))
            
            self.gpr.fit(self.X, self.y)