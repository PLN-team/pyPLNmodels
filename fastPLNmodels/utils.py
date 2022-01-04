#!/usr/bin/env python

"""Implements all the functions needed for the other modules. 

Created on Wed Nov  17 09:39:30 2021

@author: Bastien Batardiere, Julien Chiquet and Joon Kwon
"""

__authors__ = "Bastien Batardiere, Julien Chiquet and Joon Kwon" 
# __copyright__ = 
__credits__ = ["Bastien Batardiere", "Julien Chiquet", "Joon Kwon"]
#__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Bastien Batardière"
__email__ = "bastien.batardiere@gmail.com"
__status__ = "Production"

import math

import numpy as np
import scipy.linalg as SLA 
import torch
from scipy.linalg import toeplitz
import torch.linalg as TLA
torch.set_default_dtype(torch.float64)


if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')
#device = torch.device('cpu') # have to deal with this    



class Poisson_reg():
    """Poisson regressor class. The purpose of this class is to initialize the  
    variable beta of the PLN model.
    """
    def __init__(self): 
        """No particular initialization is needed."""
        pass
       
    def fit(
            self,Y,O,covariates, Niter_max = 300,
            tol = 0.001, lr = 0.005, verbose = False):
        """Run a gradient ascent to maximize the log likelihood, using 
        pytorch autodifferentiation. The log likelihood considered is 
        the one from a poisson regression model. It is roughly the 
        same as PLN without the latent layer Z. We are only trying 
        to have a good guess of beta before doing anything. 
        
        Args: 
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            Niter_max: int, optional. The maximum number of iteration we 
                are ready to do. Default is 300. 
            tol: non negative float, optional. The tolerance criteria. 
                We will stop if the norm of the gradient is less than 
                or equal to this threshold. Default is 0.001. 
            lr: positive float, optional. Learning rate for the gradient ascent. 
                Default is 0.005.
            verbose: bool. If True, will print some stats.  

        Returns : None. Update the parameter beta. You can access it 
                by calling self.beta. 
        """
        # Initialization of beta of size (d,p) 
        beta = torch.rand((covariates.shape[1], Y.shape[1]), device = device, requires_grad = True)
        optimizer = torch.optim.Rprop([beta], lr = lr)
        i = 0
        grad_norm = 2*tol  # Criterion
        while i<Niter_max and  grad_norm > tol :
            loss = -compute_l(Y, O, covariates, beta)
            loss.backward()
            optimizer.step()
            grad_norm = torch.norm(beta.grad)
            beta.grad.zero_()
            i+=1
            if verbose : 
                if i % 10 == 0 : 
                    print('log like : ', -loss)
                    print('grad_norm : ', grad_norm)
        if verbose :
            if i < Niter_max : 
                print('Tolerance reached in {} iterations'.format(i))
            else : 
                print('Maxium number of iterations reached')
        self.beta = beta 

        
def init_C(Y, O, covariates, beta, q):
    """Inititalization for C for the PLN model. We get a first 
    guess for Sigma that is easier to estimate and then takes 
    the q largest eigenvectors to get C.
    Args : 
        Y: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covarites: torch.tensor. Covariates, size (n,d)
        beta: torch.tensor of size (d,p)
        q: int. The dimension of the latent space, i.e. the reducted dimension. 
    Returns : 
        torch.tensor of size (p,q). The initialization of C. 
    """
    # get a guess for Sigma
    Sigma_hat = init_Sigma(Y, O, covariates, beta).detach()
    # taking the q largest eigenvectors
    C = C_from_Sigma(Sigma_hat, q)
    return C


def init_Sigma(Y, O, covariates, beta): 
    """ Initialization for Sigma for the PLN model. We take the log of Y
    (we are careful when Y=0), removed the covariates effects X@beta and 
    then do as a MLE for Gaussians samples. 
    Args : 
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            beta: torch.tensor of size (d,p)
    Returns : torch.tensor of size (p,p). 
    """
    # Take the log of Y, and be careful when Y = 0. If Y = 0, 
    # then we set the log(Y) as 0. 
    log_Y = torch.log(Y + (Y==0)) # we should set the log of Y 
                                  # as -2 or something like this when Y=0 
    # we remove the mean so that we see only the covariances
    

    
    log_Y_c = log_Y - torch.matmul(covariates.unsqueeze(1),beta.unsqueeze(0)).squeeze()
    # MLE in a Gaussian setting 
    Sigma_hat = torch.mean(
                    torch.matmul(log_Y_c.unsqueeze(2), 
                                 log_Y_c.unsqueeze(1)),
                                             axis = 0)
    return Sigma_hat


def init_M(Y, O, covariates, beta,C,  N_iter_max, lr, eps = 7e-3):
    '''Initialization for the variational parameter M. Basically, we take 
    the mode of the log_posterior as initialization.

    Args: 
        Y: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        beta: torch.tensor of size (d,p)
        N_iter: int. The maximum number of iteration you are ready to 
            do to find the mode. 
        lr: positive float. The learning rater of the optimizer. A good 
    '''
    W = torch.randn(Y.shape[0], C.shape[1], device = device)
    W.requires_grad_(True)
    optimizer = torch.optim.Rprop([W], lr = lr)
    criterion = 2*eps
    old_W = torch.clone(W)
    keep_condition = True
    i = 0 
    while  i < N_iter_max and keep_condition: 
        loss = -torch.mean(batch_log_P_WgivenY(Y, O, covariates,  W, C, beta) )
        loss.backward()
        optimizer.step()
        crit = torch.max(torch.abs(W-old_W))
        optimizer.zero_grad()
        
        if crit<eps and i > 2 : 
            keep_condition = False 
        old_W = torch.clone(W)
        i+=1
    print('nb iteration to find the mode: ', i)
    return W
    
    
def compute_l(Y, O, covariates, beta):
    """Compute the log likelihood of a Poisson regression."""
    # Matrix multiplication of X and beta. 
    XB = torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    # Returns the formula of the log likelihood of a poisson regression model. 
    return torch.sum(-torch.exp(O + XB)+torch.multiply(Y,O+XB))    


def sample(O,X,true_beta):
    """Sample some data according to a poisson regression."""
    parameter = np.exp(O + X@true_beta)
    return torch.poisson(parameter)


def sigmoid(x):
    """Compute the sigmoid function of x element-wise."""
    return 1/(1+ torch.exp(-x))


def sample_PLN(Sigma, beta, O, covariates, B_zero = None, ZI = False):
    """Sample Poisson log Normal variables. If ZI is True, then the model will
    be zero inflated.
    The sample size n is the the first size of O, the number p of variables
    considered is the second size of O. The number d of covariates considered
    is the first size of beta.
    
    Args: 
        Sigma: torch.tensor of size (p,p). Should be symmetric. 
        beta: torch.tensor of size (d,p). 
        0: torch.tensor. Offset, size (n,p)
        covariates : torch.tensor. Covariates, size (n,d)
        B_zero: torch.tensor of size (d,p), optional. the default is None. 
            If you set ZI to True, it will raise an error if you don't set 
            a value. 
        ZI: bool. If True, the model will be Zero Inflated. Default is False.
    Returns : 
        Y: torch.tensor of size (n,p), the count variables. 
        Z: torch.tensor of size (n,p), the gaussian latent variables.
        ksi: torch.tensor of size (n,p), the bernoulli latent variables. 
    """
    
    n = O.shape[0]
    p = Sigma.shape[0]
    # Cholesky factorization. We need to take the cholesky of Sigma   
    # in order to simulate a gaussian with variance Sigma. 
    chol = torch.cholesky(Sigma).to(device)
    # taking the square root of Sigma is another possibility,
    # less stable than the cholesky factorization. 
    #root = torch.from_numpy(SLA.sqrtm(self.Sigma)).double() 

    # Matrix multiplication between gaussians and the cholesky factorization
    # of Sigma, giving a gaussian of mean 0 and covariance Sigma. 
    Z = torch.mm(torch.randn(n,p, device = device),chol.T)
    parameter = torch.exp(O + covariates@beta + Z)
    if ZI :
        ZI_cov = covariates@B_zero
        ksi = torch.distributions.bernoulli.Bernoulli(1/(1+ torch.exp(-ZI_cov)))
    else :
        ksi = 0 
    Y = (1-ksi)*torch.poisson(parameter)
    return Y, Z, ksi


def logit_(x) : 
    """logit function"""
    return torch.log(x/(1-x))


def logit(x):
    """logit function. If x is too close from 1, we set the result to 0. 
    performs logit element wise."""
    return torch.nan_to_num(torch.log(x/(1-x)), nan =0, neginf = 0, posinf = 0)


def build_block_Sigma(p,block_size): 
    """Build a matrix per block of size (p,p). There will be p//block_size+1
    blocks of size block_size. The first p//block_size ones will be the same 
    size. The last one will have a smaller size (size (0,0) if p%block_size = 0). 
    Args: 
        p: int. 
        block_size: int. Should be lower than p. 
    Returns: a torch.tensor of size (p,p) and symmetric. 
    """
    #np.random.seed(0)
    k = p//block_size # number of matrices of size p//block_size.  
    alea = np.random.randn(k+1)**2+1# will multiply each block by some random quantities 
    Sigma = np.zeros((p,p))
    last_block_size = p-k*block_size
    # We need to form the k matrics of size p//block_size  
    for i in range(k): 
        Sigma[i*block_size : (i+1)*block_size ,i*block_size : 
              (i+1)*block_size] = alea[i]*toeplitz(0.7**np.arange(block_size))
    # Last block matrix.
    if last_block_size >0 :
        Sigma[-last_block_size:,-last_block_size:] = alea[k]*toeplitz(
                                0.7**np.arange(last_block_size))
    return Sigma


def C_from_Sigma(Sigma,q): 
    """Get the best matrix of size (p,q) when Sigma is of size (p,p). i.e. reduces norm(Sigma-C@C.T)
    Args : 
        Sigma: np.array of size (p,p). Should be positive definite and symmetric.
        q: int. The number of columns you want in your matrix C. 
        
    Returns: 
        C_reduct: np.array of size (p,q) that contains the q eigenvectors with largest eigenvalues. 
    """
    w,v = TLA.eigh(Sigma) # get the eigenvaluues and eigenvectors
    C_reduct = v[:,-q:]@torch.diag(torch.sqrt(w[-q:])) # we take only the q best. 
    return C_reduct


def log_stirling(n_):
    """Compute log(n!) even for n large. We use the Stirling formula to avoid 
    numerical infinite values of n!. 
    Args: 
         n_: torch.tensor of any size.  
    Returns: 
        An approximation of log(n!) element-wise. 
    """
    n = torch.clone(n_) #clone the tensor for precaution
    n+= (n==0) # replace the 0 with 1. It changes nothing since 0! = 1! 
    return torch.log(torch.sqrt(2*np.pi*n))+n*torch.log(n/math.exp(1)) #Stirling formula

def MSE(tens): 
    """Compute the mean of the squared (element-wise) tensor."""
    return torch.mean(tens**2)

def RMSE(tens): 
    """Compute the root mean of the squared (element-wise) tensor."""
    return torch.sqrt(torch.mean(tens**2))

def refined_MSE(sparse_tensor): 
    '''Compute the MSE of a tensor but only on the 9 largest diagonals.'''
    diag = torch.diagonal(sparse_tensor**2, offset = 0)
    for i in range(1,5): 
        diag = torch.cat((diag, torch.diagonal(sparse_tensor**2, offset = i)))
        diag = torch.cat((diag, torch.diagonal(sparse_tensor**2, offset = -i)))
    return torch.mean(diag)

def batch_log_P_WgivenY(Y_b, O_b, covariates_b, W, C, beta): 
    '''Compute the log posterior of the PLN model. Compute it either 
    for W of size (N_samples, N_batch,q) or (batch_size, q). We need to have
    both cases since we do it for both cases. 
    Args : 
        Y_b : torch.tensor of size (batch_size, p) 
        covariates_b : torch.tensor of size (batch_size, d) or (d)     
    Returns: torch.tensor of size (N_samples, batch_size) or (batch_size).  
    '''
    length = len(W.shape)
    q = W.shape[-1]
    if length == 2 : 
        CW = torch.matmul(C.unsqueeze(0),W.unsqueeze(2)).squeeze()
    elif length == 3 : 
        CW = torch.matmul(C.unsqueeze(0).unsqueeze(1), W.unsqueeze(3)).squeeze()
    A_b = O_b + CW + covariates_b@beta
    first_term = -q/2*math.log(2*math.pi)-1/2*torch.norm(W, dim = -1)**2
    second_term = torch.sum(-torch.exp(A_b)   + A_b*Y_b - log_stirling(Y_b) , axis = -1) 
    return first_term + second_term


