# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:30:02 2021

@author: Gabriel A. Morales Ruiz
"""

import numpy as np
import cvxpy as cp
from sklearn import metrics

class L1e_SVR :
    
    def __init__(self, c = 1, e = 1, k_type = "rbf", sigma = 2) :
        """
        Constructor for SVR object.

        Parameters
        ----------
        c : float, optional
            Regularization hyperparameter. Must be greater than 0. 
            The default is 1.
        e : float, optional
            Epsilon hyperparameter: Radius of the margin. 
            Must be greater than 0. The default is 1.
        k_type : str, optional
            Kernel type. Options are "linear" and "rbf". 
            The default is "rbf".
        sigma : float, optional
            Hyperparameter used on rbf kernel. The default is 2.

        Returns
        -------
        None.

        """
        self.c = c
        self.e = e
        self.k_type = k_type
        self.sigma = sigma
        
        self.X = None
        self.b = None
        self.y = None
        self.beta = None
        
    def Kernel(self, xk, xl) :
        """
        phi(xk)phi(xl)^T
        Can be either linear or rbf
        Parameters
        ----------
        xk : np.matrix
            Feature vector (new features when using to predict)
        xl : np.matrix
            Feature vector (fitted X)

        Raises
        ------
        AssertionError
            Error when kernel type is unknown

        Returns
        -------
        np.matrix
            Result of operation

        """
        if self.k_type == "linear" :
            return xk @ xl.T
        elif self.k_type == "rbf" :
            N1 = xk.shape[0]
            N2 = xl.shape[0]
            kernel = np.zeros((N1, N2))
            for k in range(N1) :
                for l in range(N2) :
                    kernel[k, l] = np.exp(\
                        - np.linalg.norm(xk[k, :] - xl[l, :])**2 / \
                            (2*self.sigma**2))
            return kernel
        else :
            raise AssertionError("Unknown kernel type.")

    
    def fit(self, X, y) :
        """
        Trains the SVR with the input parameters

        Parameters
        ----------
        X : np.matrix
            Features
        y : np.matrix
            Known output

        Returns
        -------
        None.

        """
        K = self.Kernel(X, X)
        
        N = X.shape[0]
        beta = cp.Variable((N, 1))
        v1 = np.ones((N, 1))
        ev = v1*self.e
        
        dual = cp.quad_form(beta, K)/2 + ev.T @ cp.abs(beta) - y.T @ beta
        constraints = [v1.T @ beta == 0,
                       beta <= self.c,
                       beta >= -self.c]
        cp.Problem(cp.Minimize(dual), constraints).solve(solver = "ECOS")
        beta = np.matrix(beta.value)
        
        # Support vectors
        sv = abs(beta) > 1e-5
        self.beta = beta[sv].T
        n_sv = self.beta.shape[0]
        self.X = X[np.repeat(sv, X.shape[1], axis=1)].reshape(n_sv, X.shape[1])
        
        # Compute b
        sb = np.logical_and( abs(beta) > 1e-5, abs(beta) < self.c )
        beta_sb = beta[sb].T
        n_sb = beta_sb.shape[0]
        y_sb = y[sb].T
        K_sb = K[sb*sb.T].reshape(n_sb, n_sb)
        e_sb = np.sign(beta_sb)*self.e
        self.b = np.mean(y_sb - (K_sb*beta_sb) + e_sb)
        
    def predict(self, X_new) :
        """
        Uses the fitted model to predict an output with input features

        Parameters
        ----------
        X_new : np.matrix
            Features

        Returns
        -------
        np.matrix
            Prediction based on input.

        """
        K = self.Kernel(X_new, self.X)
        return (sum(np.multiply(self.beta, K.T)) + self.b).T
    
    def mse(self, X_new, y_new) :
        """
        Calculates the mean squared error

        Parameters
        ----------
        X_new : np.matrix
            Input to predict.
        y_new : np.matrix
            Correct result

        Returns
        -------
        float
            Mean squared error of the prediction - the real value.

        """
        y_pred = self.predict(X_new)
        mse = np.square(y_pred - y_new).mean(axis=0)
        return mse[0, 0]

    def score(self, X_new, y_new) :
        """
        Calculates R^2 score

        Parameters
        ----------
        X_new : np.matrix
            Input to predict
        y_new : TYPE
            Correct result

        Returns
        -------
        float
            R^2 score.

        """
        y_pred = self.predict(X_new)
        return metrics.r2_score(y_pred, y_new)
