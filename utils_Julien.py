import numpy as np
from scipy import linalg
from scipy import optimize
import torch
torch.set_default_dtype(torch.float64)
import scipy.linalg as SLA 
from scipy.linalg import toeplitz
import math
torch.set_default_dtype(torch.float64)



def log_factorial(A):
    """ 
        function to compute an approximation of the sum of the log factorial for a matrix argument
    """ 
    v = np.where(A == 0, 1, A)
    return np.sum(v * np.log(v) - v + np.log(8 * v ** 3 + 4 * v ** 2 + v + 1/30)/6 + np.log(np.pi)/2)

def log_factorial_component_wise(A):
    v = np.where(A == 0, 1, A)
    return v * np.log(v) - v + np.log(8 * v ** 3 + 4 * v ** 2 + v + 1/30)/6 + np.log(np.pi)/2

def moments_estimate(Y, O = None, min_var = 1e-3):
    n = Y.shape[0]
    if O is None: O = np.zeros_like(Y)
    EY  = np.mean(Y   , 0)
    EY2 = np.mean(Y**2, 0)
    s = (EY2 - EY)/(EY**2)
    s = np.maximum(np.log(np.where(s == 0, 1, s)), min_var)
    M0 = np.log(EY) - 0.5 * s  - O
    S0 = np.outer(np.ones(n), s)
    return M0, S0

def format_data(counts, covariates = None, offsets = None) :
    O = np.zeros_like(counts) if offsets is None else offsets
    X = np.ones([counts.shape[0], 1]) if covariates is None else covariates
    return {
        "Y"     : counts,
        "O"     : O,
        "X"     : X,
        "projX" : X @ linalg.inv(X.T @X) @ X.T,
        "KY"    : log_factorial(counts)
    }

def _variational_vec2mat(x, n, p):
    """ extract variational paramters in matrix form"""
    return x[0:(n*p)].reshape(n, p), x[(n*p):(2*n*p)].reshape(n, p)

def _variational_mat2vec(M, S):
    return np.concatenate((M.flatten(), S.flatten()))
    
def _variational_model_params(M, S, projX):
    """replace mu and Sigma per their explicit estimator"""
    mu = projX @ M
    S2 = S * S
    Sigma = (M - mu).T @ (M - mu) / M.shape[0] + np.diag(S2.mean(0))
    return mu, Sigma

def _laplace_model_params(Z, projX, threshold = 1e-5):
    """replace mu and Sigma per their explicit estimator"""
    mu = projX @ Z
    D, U = linalg.eig((Z - mu).T @ (Z - mu))
    D = D.real
    U = U.real
    D = np.where(D <= threshold, threshold, D)

    def myfunc(x, Z, D):
        return( np.sum( (x**2) * np.exp(Z) /(1 + x * np.exp(Z) ) , 0) - D )

    res = optimize.root(myfunc, 1/D, args = (Z, D))
    Lambda = np.diag(res.x)

    Sigma = U @ Lambda @ U.T
   
    return mu, Sigma

#from sklearn import linear_model
#def LM_estimate(X, Y, starting_var = 1e-1):
#    myLM = linear_model.LinearRegression(fit_intercept = False)
#    
#    myLM.fit(X, np.log(1 + Y))
#    residuals = np.log(1 + Y) - myLM.predict(X)
#    M0 = np.log(myLM.coef_[:,0]) + residuals
#    S0 = starting_var * np.ones_like(Y) 
#    
#    return M0, S0



