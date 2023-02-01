#!/usr/bin/env python
"""Implements all the functions needed for the other modules.

Created on Wed Nov  17 09:39:30 2021

@author: Bastien Batardiere, Julien Chiquet and Joon Kwon
"""

__authors__ = "Bastien Batardiere, Julien Chiquet and Joon Kwon"
# __copyright__ =
__credits__ = ["Bastien Batardiere", "Julien Chiquet", "Joon Kwon"]
# __license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Bastien Batardière"
__email__ = "bastien.batardiere@gmail.com"
__status__ = "Production"

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg as TLA
from scipy.linalg import toeplitz

# torch.set_default_dtype(torch.float64)
# O is not doing anything in the initialization of Sigma. should be fixed.

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device('cpu') # have to deal with this


class PLNPlotArgs:
    def __init__(self, window):
        self.window = window
        self.running_times = list()
        self.criterions = [1] * window
        self.ELBOs_list = list()


    @property
    def iteration_number(self):
        return len(self.ELBOs_list)

    def show_loss(self, ax=None, savefig=False, nameDoss=""):
        """Show the ELBO of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The ELBO will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'name_doss': str. The name of the file the graphic will be saved to.
                Default is 'fastPLNPCA_ELBO'.
        returns: None but displays the ELBO.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(
            self.running_times,
            -np.array(self.ELBOs_list),
            label="Negative ELBO",
        )
        ax.set_title("Negative ELBO. Best ELBO = " +
                     str(np.round(self.ELBOs_list[-1], 6)))
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("ELBO")
        ax.legend()
        # save the graphic if needed
        if savefig:
            plt.savefig(nameDoss)

    def show_stopping_criterion(self, ax=None, savefig=False, nameDoss=""):
        """Show the criterion of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The criterion will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'name_doss': str. The name of the file the graphic will be saved to.
                Default is 'fastPLN_criterion'.
        returns: None but displays the criterion.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.running_times[self.window:],
                self.criterions[self.window:],
                label="Delta")
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Delta")
        ax.set_title("Increments")
        ax.legend()
        # save the graphic if needed
        if savefig:
            plt.savefig(nameDoss)


class PoissonRegressor:
    def fit(self,
            Y,
            covariates,
            O,
            Niter_max=300,
            tol=0.001,
            lr=0.005,
            verbose=False):
        """Run a gradient ascent to maximize the log likelihood, using
        pytorch autodifferentiation. The log likelihood considered is
        the one from a poisson regression model. It is roughly the
        same as PLN without the latent layer Z.

        Args:
            Y: torch.tensor. Counts with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            Niter_max: int, optional. The maximum number of iteration.
                Default is 300.
            tol: non negative float, optional. The tolerance criteria.
                Will stop if the norm of the gradient is less than
                or equal to this threshold. Default is 0.001.
            lr: positive float, optional. Learning rate for the gradient ascent.
                Default is 0.005.
            verbose: bool, optional. If True, will print some stats.

        Returns : None. Update the parameter beta. You can access it
                by calling self.beta.
        """
        # Initialization of beta of size (d,p)
        beta = torch.rand((covariates.shape[1], Y.shape[1]),
                          device=device,
                          requires_grad=True)
        optimizer = torch.optim.Rprop([beta], lr=lr)
        i = 0
        gradNorm = 2 * tol  # Criterion
        while i < Niter_max and gradNorm > tol:
            loss = -poissreg_loglike(Y,  covariates,O, beta)
            loss.backward()
            optimizer.step()
            gradNorm = torch.norm(beta.grad)
            beta.grad.zero_()
            i += 1
            if verbose:
                if i % 10 == 0:
                    print("log like : ", -loss)
                    print("gradNorm : ", gradNorm)
        if verbose:
            if i < Niter_max:
                print("Tolerance reached in {} iterations".format(i))
            else:
                print("Maxium number of iterations reached")
        self.beta = beta


def init_Sigma(Y,  covariates,O, beta):
    """ Initialization for Sigma for the PLN model. Take the log of Y
    (careful when Y=0), remove the covariates effects X@beta and
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
    log_Y = torch.log(Y + (Y == 0) * math.exp(-2))
    # we remove the mean so that we see only the covariances
    log_Y_centered = log_Y - \
        torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    # MLE in a Gaussian setting
    n = Y.shape[0]
    Sigma_hat = 1 / (n - 1) * (log_Y_centered.T) @ log_Y_centered

    return Sigma_hat


def init_C(Y,  covariates,O, beta, q):
    """Inititalization for C for the PLN model. Get a first
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
    Sigma_hat = init_Sigma(Y,  covariates,O, beta).detach()
    # taking the q largest eigenvectors
    C = C_from_Sigma(Sigma_hat, q)
    return C


def init_M(Y, covariates,O, beta, C, N_iter_max, lr, eps=7e-3):
    """Initialization for the variational parameter M. Basically,
    the mode of the log_posterior is computed.

    Args:
        Y: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        beta: torch.tensor of size (d,p)
        N_iter_max: int. The maximum number of iteration in
            the gradient ascent.
        lr: positive float. The learning rate of the optimizer.
        eps: positive float, optional. The tolerance. The algorithm will stop if
            the maximum of |W_t-W_{t-1}| is lower than eps, where W_t
            is the t-th iteration of the algorithm.This parameter
            changes a lot the resulting time of the algorithm. Default is 9e-3.
    """
    W = torch.randn(Y.shape[0], C.shape[1], device=device)
    W.requires_grad_(True)
    optimizer = torch.optim.Rprop([W], lr=lr)
    crit= 2 * eps
    old_W = torch.clone(W)
    keep_condition = True
    i = 0
    while i < N_iter_max and keep_condition:
        loss = -torch.mean(log_PW_given_Y(Y, covariates,O, W, C, beta))
        loss.backward()
        optimizer.step()
        crit = torch.max(torch.abs(W - old_W))
        optimizer.zero_grad()
        if crit < eps and i > 2:
            keep_condition = False
        old_W = torch.clone(W)
        i += 1
    print("nb iteration to find the mode: ", i)
    return W


def poissreg_loglike(Y, covariates,O, beta):
    """Compute the log likelihood of a Poisson regression."""
    # Matrix multiplication of X and beta.
    XB = torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    # Returns the formula of the log likelihood of a poisson regression model.
    return torch.sum(-torch.exp(O + XB) + torch.multiply(Y, O + XB))


def sigmoid(x):
    """Compute the sigmoid function of x element-wise."""
    return 1 / (1 + torch.exp(-x))


def sample_PLN(C, beta, covariates,O, B_zero=None):
    """Sample Poisson log Normal variables. If B_zero is not None, the model will
    be zero inflated.

    Args:
        C: torch.tensor of size (p,q). The matrix C of the PLN model
        beta: torch.tensor of size (d,p). Regression parameter. 
        0: torch.tensor of size (n,p0. Offsets. 
        covariates : torch.tensor of size (n,d). Covariates. 
        B_zero: torch.tensor of size (d,p), optional. If B_zero is not None,
             the ZIPLN model is chosen, so that it will add a 
             Bernouilli layer. Default is None.
    Returns :
        Y: torch.tensor of size (n,p), the count variables.
        Z: torch.tensor of size (n,p), the gaussian latent variables.
        ksi: torch.tensor of size (n,p), the bernoulli latent variables 
        (full of zeros if B_zero is None).
    """

    n = O.shape[0]
    q = C.shape[1]
    Z = torch.mm(torch.randn(n, q, device=device), C.T) + covariates @ beta
    parameter = torch.exp(O + Z)
    if B_zero is not None:
        print("ZIPLN is sampled")
        ZI_cov = covariates @ B_zero
        ksi = torch.bernoulli(1 / (1 + torch.exp(-ZI_cov)))
    else:
        ksi = 0
    Y = (1 - ksi) * torch.poisson(parameter)
    return Y, Z, ksi


def logit(x):
    """logit function. If x is too close from 1, we set the result to 0.
    performs logit element wise."""
    return torch.nan_to_num(torch.log(x / (1 - x)), nan=0, neginf=0, posinf=0)


def build_block_Sigma(p, block_size):
    """Build a matrix per block of size (p,p). There will be p//block_size+1
    blocks of size block_size. The first p//block_size ones will be the same
    size. The last one will have a smaller size (size (0,0) if p%block_size = 0).
    Args:
        p: int.
        block_size: int. Should be lower than p.
    Returns: a torch.tensor of size (p,p) and symmetric.
    """
    # np.random.seed(0)
    k = p // block_size  # number of matrices of size p//block_size.
    # will multiply each block by some random quantities
    alea = np.random.randn(k + 1)**2 + 1
    Sigma = np.zeros((p, p))
    last_block_size = p - k * block_size
    # We need to form the k matrics of size p//block_size
    for i in range(k):
        Sigma[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) *
              block_size] = alea[i] * toeplitz(0.7**np.arange(block_size))
    # Last block matrix.
    if last_block_size > 0:
        Sigma[-last_block_size:, -last_block_size:] = alea[k] * toeplitz(
            0.7**np.arange(last_block_size))
    return Sigma


def C_from_Sigma(Sigma, q):
    """Get the best matrix of size (p,q) when Sigma is of
    size (p,p). i.e. reduces norm(Sigma-C@C.T)
    Args :
        Sigma: torch.tensor of size (p,p). Should be positive definite and symmetric.
        q: int. The number of columns wanted for C

    Returns:
        C_reduct: torch.tensor of size (p,q) containing the q eigenvectors with largest eigenvalues.
    """
    w, v = TLA.eigh(Sigma)  # Get the eigenvaluues and eigenvectors
    # Take only the q largest
    C_reduct = v[:, -q:] @ torch.diag(torch.sqrt(w[-q:]))
    return C_reduct


def init_beta(Y, covariates, O):
    poiss_reg = PoissonRegressor()
    poiss_reg.fit(Y, covariates,O)
    return torch.clone(poiss_reg.beta.detach()).to(device)


def log_stirling(n):
    """Compute log(n!) even for n large. We use the Stirling formula to avoid
    numerical infinite values of n!.
    Args:
         n: torch.tensor of any size.
    Returns:
        An approximation of log(n_!) element-wise.
    """
    n_ = n + \
        (n == 0)  # Replace the 0 with 1. It doesn't change anything since 0! = 1!
    return torch.log(torch.sqrt(
        2 * np.pi * n_)) + n_ * torch.log(n_ / math.exp(1))  # Stirling formula


def log_PW_given_Y(Y_b, covariates_b,O_b, W, C, beta):
    """Compute the log posterior of the PLN model. Compute it either
    for W of size (N_samples, N_batch,q) or (batch_size, q). Need to have
    both cases since it is done for both cases after. Please the mathematical 
    description of the package for the formula. 
    Args :
        Y_b : torch.tensor of size (batch_size, p)
        covariates_b : torch.tensor of size (batch_size, d) or (d)
    Returns: torch.tensor of size (N_samples, batch_size) or (batch_size).
    """
    length = len(W.shape)
    q = W.shape[-1]
    if length == 2:
        CW = torch.matmul(C.unsqueeze(0), W.unsqueeze(2)).squeeze()
    elif length == 3:
        CW = torch.matmul(C.unsqueeze(0).unsqueeze(1),
                          W.unsqueeze(3)).squeeze()

    A_b = O_b + CW + covariates_b @ beta
    first_term = -q / 2 * math.log(2 * math.pi) - \
        1 / 2 * torch.norm(W, dim=-1) ** 2
    second_term = torch.sum(-torch.exp(A_b) + A_b * Y_b - log_stirling(Y_b),
                            axis=-1)
    return first_term + second_term


def plot_list(myList, label, ax=None):
    if ax == None:
        ax = plt.gca()
    ax.plot(np.arange(len(myList)), myList, label=label)


def trunc_log(x, eps=1e-16):
    y = torch.min(torch.max(x, torch.tensor([eps])), torch.tensor([1 - eps]))
    return torch.log(y)


def getOFromSumOfY(Y):
    sumOfY = torch.sum(Y, axis = 1)
    return sumOfY.repeat((Y.shape[1], 1)).T

