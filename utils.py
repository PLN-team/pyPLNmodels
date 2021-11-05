#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:30 2019

@author: jchiquet
"""

import numpy as np
from scipy import linalg
from scipy import optimize
import torch
torch.set_default_dtype(torch.float64)
import scipy.linalg as SLA 
from scipy.linalg import toeplitz
import math
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')


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




class Poisson_reg():
    '''
    Poisson regressor class. The purpose of this class is to initialize the PLN model.
    '''
    
    def __init__(self): 
        pass
    

    
    def fit(self,O,X,Y, Niter_max = 300, tol = 0.1, lr = 0.00008,  verbose = False): 
        '''
        We run a gradient ascent to maximize the log likelihood. We do this by hand : we compute the gradient ourselves. 
        The log likelihood considered is the one from a poisson regression model. It is the same as PLN without the latent layer Z. 
        We are only trying to have a good guess of beta before doing anything. 
        
        args : 
                '0' : offset, size (n,p)
                'X' : covariates, size (n,d)
                'Y' : samples , size (n,p)
                'Niter_max' :int  the number of iteration we are ready to do 
                'tol' : float. the tolerance criteria. We will stop if the norm of the gradient is less than 
                       or equal to this threshold
                'lr' : float. learning rate for the gradient ascent
                'verbose' : bool. if True, will print some stats on the 
                
        returns : None but update the parameter beta 
        '''
        
        #we initiate beta 
        beta = torch.rand(X.shape[0])
        i = 0
        grad_norm = 2*tol
        while i<Niter_max and  grad_norm > tol : # condition to keep going
            grad = grad_poiss_beta(O,X,Y,beta) # computes the gradient 
            grad_norm = torch.norm(grad) 
            beta += lr*grad_poiss_beta(O,X,Y,beta)# update beta 
            i+=1
            
            # some stats if we want some 
            if verbose == True : 
                if i % 10 == 0 : 
                    print('log likelihood  : ', compute_l(0,X,Y,beta))
                    print('Gradient norm : ', grad_norm)
        if i < Niter_max : 
            print('---------------------Tolerance reachedin {} iterations'.format(i))
        else : 
            print('---------------------Maximum number of iterations reached')
        print('----------------------Gradient norm : ', grad_norm)
        self.beta = beta # save beta 
        
        
    def fit_torch(self,O,X,Y, Niter_max = 300, tol = 0.1, lr = 0.005, verbose = False): 
        '''
        Does exaclty the same as fit() but uses autodifferentiation of pytorch. 
        '''
        
        beta = torch.rand((X.shape[1], Y.shape[1]), requires_grad = True).to(device)
        
        optimizer = torch.optim.RMSprop([beta], lr = lr)
        i = 0
        grad_norm = 2*tol
        while i<Niter_max and  grad_norm > tol :
            loss = -compute_l(O,X,Y,beta)
            loss.backward()
            optimizer.step()
            grad_norm = torch.norm(beta.grad)
            beta.grad.zero_()
            i+=1
            if verbose == True : 
                if i % 10 == 0 : 
                    print('log like : ', -loss)
                    print('grad_norm : ', grad_norm)
        if verbose :
            if i < Niter_max : 
                print('-------------------Tolerance reached in {} iterations'.format(i))
            else : 
                print('-------------------Maxium number of iterations reached')
        self.beta = beta 
    

def grad_poiss_beta(O,X,Y,beta): 
    
    return torch.sum(-torch.multiply(X,torch.exp(O+X@beta).reshape(-1,1))+torch.multiply(Y.reshape(-1,1),X),dim = 0)
    
def compute_l(O,X,Y,beta):
    XB = torch.matmul(X.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    return torch.sum(-torch.exp(O + XB)+torch.multiply(Y,O+XB))    
    
def sample(O,X,true_beta):
        parameter = np.exp(O + X@true_beta)
        return torch.poisson(parameter)
    
def sigmoid(x):
    return 1/(1+ torch.exp(-x))


class sample_PLN(): 
    '''
    simple class to sample some variables with the PLN model. 
    The main method is the sample one, however we can also plot the data calling the plot_Y method. 
    The method conditional prior should not be used and have not been tested properly. 
    '''
    
    def __init__(self, ZI = False):
        self.ZI = ZI
        pass 
    
    def sample(self, Sigma, beta, O, covariates, B_zero = None ): 
        '''
        sample Poisson log Normal variables. 
        The number of samples is the the first size of O, the number of species
        considered is the second size of O
        The number of covariates considered is the first size of beta. 
        '''
        self.Sigma = Sigma # unknown parameter in practice
        self.beta = beta #unknown parameter in practice
        
        self.O = O 
        self.covariates = covariates
        
        self.n = self.O.shape[0]
        self.p = self.Sigma.shape[0]
        chol = torch.cholesky(self.Sigma)
        #root = torch.from_numpy(SLA.sqrtm(self.Sigma)).double()
        self.Z = torch.mm(torch.randn(self.n,self.p),chol.T)
        parameter = np.exp(self.O + self.covariates@self.beta + self.Z.numpy())
        ZI_cov = self.covariates@B_zero
        if self.ZI : 
            ksi = np.random.binomial(1,1/(1+ np.exp(-ZI_cov)))
        else :
            ksi = 0 
        self.Y = (1-ksi)*np.random.poisson(lam = parameter)
        return self.Y, self.Z, ksi

def logit_(x) : 
    return torch.log(x/(1-x))

def logit(x):
    y = x + (x==0)*0.5
    return torch.nan_to_num(torch.log(x/(1-x)), nan =0, neginf = 0, posinf = 0)

    
    
def M_x(t,mu,Sigma): 
    return np.exp(mu@t + 1/2*t@Sigma@t)

def build_block_Sigma(p,block_size): 
    '''
    build a matrix per block of size (p,p). There will be p//block_size+1 blocks of size block_size.
    The first p//k ones will be the same size. The last one will be different (size (0,0) if p%block_size = 0)
    '''
    np.random.seed(0)
    k = p//block_size
    alea = np.random.randn(k+1)**2+1# will multiply each block by some random quantities 
    Sigma = np.zeros((p,p))
    last_block_size = p-k*block_size
    #block_size,last_block_size = p//k, p%k
    for i in range(k): 
        Sigma[i*block_size : (i+1)*block_size ,i*block_size : (i+1)*block_size] = alea[i]*toeplitz(0.7**np.arange(block_size))
    if last_block_size >0 :
        Sigma[-last_block_size:,-last_block_size:] = alea[k]*toeplitz(0.7**np.arange(last_block_size))
    return Sigma+0.1*toeplitz(0.95**np.arange(p))


def C_from_Sigma(Sigma,q): 
    ''' 
    get the best matrix of size (p,q) when Sigma is of size (p,p). i.e. reduces norm(Sigma-C@C.T)
    args : 
        Sigma : np.array of size (p,p). Should be positive definite and symmetric.
        q : int. The number of columns you want in your matrix C. 
        
    returns : C_reduct : np.array of size (p,q) that contains the q eigenvectors with largest eigenvalues. 
    '''
    w,v = SLA.eigh(Sigma) # get the eigenvaluues and eigenvectors
    C_reduct = v[:,-q:]@np.diag(np.sqrt(w[-q:])) # we take only the q best. 
    return C_reduct


def log_stirling(n_):
    '''
    this function computes log(n!) even for n large. We use the Stirling formula to avoid 
    numerical infinite values of n!. It can also take tensors.
    
    args : 
         n_ : tensor. 
    return : an approximation of log(n!)
    '''
    n = torch.clone(n_) #clone the tensor by precaution
    n+= (n==0) # replace the 0 with 1. It changes nothing since 0! = 1! 
    return torch.log(torch.sqrt(2*np.pi*n))+n*torch.log(n/math.exp(1)) #Stirling formula

def MSE(tens): 
    '''
    computes the Mean Squared Error of a torch.tensor
    '''
    return torch.mean(tens**2)