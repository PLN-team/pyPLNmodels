#!/usr/bin/env python

"""Implement all the model related to the PLN model, such as:
    - Variational model for PLN: fastPLN. It is very fast, but
        can't do dimension reduction. We can only infer Sigma that
        has size (p,p).
    - Variational model for PLN-PCA: fastPLNPCA
    - Importance Sampling based model for PLN-PCA. Relatively slow
        compared to fastPLN and fastPLNPCA. However, we don't do any approximation
        and infers the MLE.


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
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.linalg as TLA
from tqdm import tqdm

from .utils import C_from_Sigma, Poisson_reg, batch_log_P_WgivenY, init_C
from .utils import init_M, init_Sigma, log_stirling
from .VRA import SAGARAD, SAGRAD, SVRGRAD
 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Device ', device)


def log_likelihood(Y, O, covariates, C, beta, acc=0.002,
                   N_iter_mode=1000, lr_mode=0.1):
    """Estimate the log likelihood of C and beta given Y,O,covariates.
    The process is a little bit complicated since it needs to find
    the mode of the posterior in order to sample the right Gaussians.

    Args:
        Y: pd.DataFrame of size (n, p). The counts
        O: pd.DataFrame of size (n,p). The offset
        covariates: pd.DataFrame of size (n,p).
        C: torch.tensor of size (p,q)
        beta: torch.tensor of size (d,p)
        acc: positive float, optional. The accuracy wanted. Basically,
            will sample 1/acc gaussians to estimate the likelihood.
            Default is 0.002.
        N_iter_mode: int, optional. The maximum number of iteration to do to
            find the mode of the posterior. Default is 1000. Should
            not be lower since we need a very accurate mode.
        lr_mode: positive float, optional. The learning rate of the
            optimizer that finds the mode. Default is 0.1
    Returns:
        The approximate likelihood of the whole dataset.
    """
    q = C.shape[1]
    # Initialize an IMPS_PLN model that will estimate the log likelihood.
    model = IMPS_PLN(q)
    model.nb_trigger = 5
    model.j = 1
    model.init_data(Y, O, covariates, good_init=False)
    model.Y_b, model.O_b, model.covariates_b = model.Y, model.O, model.covariates
    model.C_mean = C
    model.beta_mean = beta
    log_like = model.compute_best_loglike(acc, N_iter_mode, lr_mode)
    del(model)
    return log_like


def sample_gaussians(N_samples, mean, sqrt_Sigma):
    '''Sample some gaussians with the right mean and variance.
    Be careful, we ask for the square root of Sigma, not Sigma.

    Args:
         N_samples : int. the number of samples wanted.
         mean : torch.tensor of size (n_batch,q)
         sqrt_Sigma : torch.tensor or size (batch_size, q, q). The square roots matrices
             of the covariance matrices. (e.g. if you want to sample a gaussian with
             covariance matrix A, you need to give the square root of A in argument.

    Returns:
        W: torch.tensor of size (N_samples, batch_size,q). It is a vector
        of N_samples gaussian of dimension mean.shape. For each  1< i< N_sample,
        1<k< n_batch , W[i,k] is a gaussian with mean mean[k,:] and variance
        sqrt_Sigma[k,:,:]@sqrt_Sigma[k,:,:].
    '''
    q = mean.shape[1]
    W_orig = torch.randn(N_samples, 1, q, 1).to(device)
    # just add the mean and multiply by the square root matrice to sample from
    # the right distribution.
    W = torch.matmul(
        sqrt_Sigma.unsqueeze(0),
        W_orig).squeeze() + mean.unsqueeze(0)
    return W


def log_gaussian_density(W, mu_p, Sigma_p):
    '''Compute the log density of a gaussian W of size
    (N_samples, n_batch, q) With mean mu_p and Sigma_p.

    Args :
        W: torch.tensor of size (N_samples, batch_size, q)
        mu_p : torch.tensor : the mean from which the gaussian has been sampled.
        Sigma_p : torch.tensor. The variance from which the gaussian has been sampled.
    Returns :
        torch.tensor. The log of the density of W, taken along the last axis.
    '''
    dim = W.shape[-1]  # dimension q
    # Constant of the gaussian density
    const = torch.sqrt((2 * math.pi)**dim * torch.det(Sigma_p))
    Wmoinsmu = W - mu_p.unsqueeze(0)
    inv_Sig = torch.inverse(Sigma_p)
    # Log density of a gaussian.
    log_d = -1 / 2 * torch.matmul(
        torch.matmul(inv_Sig.unsqueeze(0),
                     Wmoinsmu.unsqueeze(3)).squeeze().unsqueeze(2),
        Wmoinsmu.unsqueeze(3))
    return log_d.squeeze() - torch.log(const)


class PLNmodel():
    
    def __init__(self,q= None,nb_average_param= 100, nb_average_likelihood= 8): 
        self.q = q 
        self.nb_average_param = nb_average_param
        self.nb_average_likelihood = nb_average_likelihood
        
    def fit(self, Y, O, covariates, N_iter_max=500,  lr=0.1,
            optimizer=torch.optim.Rprop, tol = 1e-1,  good_init=True,
            imps_optimizer = torch.optim.Adagrad, fast = True,
             VR='SAGA', batch_size=40, acc=0.005,nb_plateau=15,
             nb_trigger=5, verbose=False): 
        self.p = Y.shape[1]
        if self. q == None:
            print('You did not put any PCs in argument, so that the number') 
            print('of PCs is arbitrarily set to the maximum value.') 
            self.q = self.p 
        if self.q > self.p: 
            raise AttributeError('The number of PCs q cant be greater than p') 
        if self.q == self.p and fast == True: 
            print('Fitting a PLN model. Number of PCs: ', self.p) 
            self.model = fastPLN()
            self.model.fit(Y, O, covariates, N_iter_max=N_iter_max, tol= tol,
            optimizer=optimizer, lr=lr, good_init=good_init, verbose=verbose)
        elif self.q < self.p and fast == True:
            print('Fitting a PLNPCA model. Number of PCs: ', self.q)
            self.model = fastPLNPCA(self.q) 
            self.model.fit(Y, O, covariates, N_iter_max=N_iter_max*10, tol=tol*1e-2, 
                           optimizer=optimizer, lr=lr, good_init=good_init, verbose=verbose) 
        else: # In this case: fast = False and we do importance sampling based inference.
            print('Fitting a PLNPCA model with importance sampling. Number of PCs: ', self.p)
            self.model = IMPS_PLN(self.q, self.nb_average_param, self.nb_average_likelihood)
            self.model.fit(Y, O, covariates, acc=acc, N_iter_max=N_iter_max*2, lr=lr,
             VR=VR, batch_size=batch_size, optimizer=imps_optimizer,
             nb_plateau=nb_plateau, nb_trigger=nb_trigger, 
             good_init=good_init, verbose=verbose)
        
            
            
            
    def plot_runtime(self):
        self.model.plot_runtime()
        
    def __str__(self):
        '''Print the model that have been fitted'''
        print(self.model)
        return ''
    
    
    def get_beta(self):
        '''Getter for beta. Returns the estimated beta'''
        return self.model.get_beta()

    def get_Sigma(self):
        '''Getter for Sigma. Returns the estimated Sigma'''
        return self.model.get_Sigma()

    def get_C(self):
        '''Getter for C.'''
        return self.model.get_C()
        
            
            
        
    


class IMPS_PLN():
    '''Maximize the likelihood of the PLN-PCA model. The main method
    is the fit() method that fits the model. Most of the others
    functions are here to support the fit method. Any value of n can be taken.
    However, q should not be greater than 40. The greater q, the
    lower the accuracy parameter should be taken.
    '''

    def __init__(self, q, nb_average_param=100, nb_average_likelihood=8):
        '''Init method. Set some global parameters of the class, such as
        the dimension of the latent space and the number of elements took
        to set an average parameter that should be more accurate.

        Args :
            q : int. The dimension of the latent layer of the PLN-PCA model.
            nb_average_param: int, optional. Will average the parameter to get
                parameters with lower variance. nb_average_param tells
                the number of parameter took to build the mean. Should
                not be changed since not very important. Default is 100.
            nb_average_likelihood: int, optional. Will average the log_likelihood
                of the model. nb_average likelihood tells the number of
                likelihood took to build the mean likelihood. Should
                not be changed since not very important. Note that this
                parameter is actually changing the algorithm (just a little bit)
                since the stopping criterion depends directly on the average likelihood.
                Default is 8.
        Returns:
            An IMPS_PLN object.
        '''
        self.q = q  # dimension of the latent space
        self.nb_average_likelihood = nb_average_likelihood
        self.nb_average_param = nb_average_param
        # bool variable to know if the object has been fitted.
        self.fitted = False

    def init_data(self, Y, O, covariates, good_init):
        '''Initialise some usefuls variables given the data.
        Also initialise C and beta, if the model has not been fitted.

        Args :
               Y: pd.DataFrame of size (n, p). The counts
               O: pd.DataFrame of size (n,p). the offset
               covariates: pd.DataFrame of size (n,p)
               good_init: bool. If True, a good initialization (not random)
                   will to be performed. Takes some time.
        Returns:
                None
        '''

        self.counter_list = [0] * self.nb_trigger  # counter for the criterion
        self.t_mode_list = list()  # store the time took to find the mode at each iteration
        self.t_grad_estim_list = list() # store the time took to estimate the gradiens.
        if self.fitted == False:
            self.crit_list = [0]  # store the criterion to plot it after.
            # Import the data. We take either pandas.DataFrames or torch.tensor
            # pandas.DataFrame
            try: 
                self.Y = torch.from_numpy(Y.values).to(device)
                self.O = torch.from_numpy(O.values).to(device)
                self.covariates = torch.from_numpy(covariates.values).to(device)
            # torch.tensor (if not torch.tensor, will launch an error after) 
            except: 
                try:   
                    self.Y = Y.to(device)
                    self.O = O.to(device)
                    self.covariates = covariates.to(device)
                except: 
                    raise ValueError('Every Y,O, covariates should be either a pandas.DataFrame or a torch.tensor')

            self.n = Y.shape[0]
            self.p = Y.shape[1]
            self.d = covariates.shape[1]
            # Initialize some lists
            self.running_times = list()  # store the running times for a nice plot
            self.log_likelihood_list = list()  # store the likelihood to plot it after
            self.last_likelihoods = list()  # store the last likelihoods in order to take
            # the mean of those likelihoods to smooth it.
            self.nb_iteration_list = list()  # store the number of iteration
            # done at each gradient ascent
            # that finds the mode.
            self.iteration_cmpt = 0  # counts the number of
            # iteration we have done
            # Tensor that will store the starting point for the
            # gradient descent finding the mode for IMPS.
            self.starting_point = torch.zeros(
                self.n, self.q, device=device, requires_grad=False)
            
            self.mode_step_sizes= torch.zeros((self.n, self.q), device = device)

            # Initialization of the last beta and C.
            self.last_betas = torch.zeros(
                self.nb_average_param, self.d, self.p)
            self.last_Cs = torch.zeros(self.nb_average_param, self.p, self.q)

            print('Intialization ...')
            if good_init:
                # Initialization for beta with a poisson regression.
                poiss_reg = Poisson_reg()
                poiss_reg.fit(self.Y, self.O, self.covariates)
                self.beta = torch.clone(poiss_reg.beta.detach()).to(device)
                # Initialization for C with an array of size (p,q) taking the q vectors associated to
                # the q largest eigenvectors of the estimated variance of
                # log(Y).
                self.C = init_C(
                    self.Y,
                    self.O,
                    self.covariates,
                    self.beta,
                    self.q).to(device)
            else:
                self.beta = torch.randn((self.n, self.p), device=device)
                self.C = torch.randn((self.p, self.q), device=device)
            print('Initalization done')

        # Mean of the last nb_average_param parameters, that are
        # supposed to be more accurate after.
        self.C_mean = torch.clone(self.C.detach()).to(device)
        self.beta_mean = torch.clone(self.beta.detach()).to(device)

    def get_batch(self, batch_size):
        '''Get the batches required to do a  minibatch gradient ascent.

        Args:
            batch_size: int. The batch size. Should be lower than n.

        Returns: A generator. Will generate n//batch_size + 1 batches of
            size batch_size (except the last one since the rest of the
            division is not always 0)
        '''
        indices = np.arange(self.n)
        # Shuffle the indices to avoid a regular path.
        np.random.shuffle(indices)
        # Set the batch size of the model to the right size
        self.batch_size = batch_size
        # get the number of batches and the size of the last one.
        nb_full_batch, last_batch_size = self.n // batch_size, self.n % batch_size
        for i in range(nb_full_batch):
            yield (self.Y[indices[i * batch_size: (i + 1) * batch_size]],
                   self.covariates[indices[i *
                                           batch_size: (i + 1) * batch_size]],
                   self.O[indices[i * batch_size: (i + 1) * batch_size]],
                   indices[i * batch_size: (i + 1) * batch_size]
                   )
        # Last batch
        # Set the batch size of the model to the right size
        if last_batch_size != 0:
            self.batch_size = last_batch_size
            yield (self.Y[indices[-last_batch_size:]],
                   self.covariates[indices[-last_batch_size:]],
                   self.O[indices[-last_batch_size:]],
                   indices[-last_batch_size:]
                   )

    def average_likelihood(self):
        '''Average the likelihood to smooth it. Do so since we can only estimate
        the likelihood, thus it is random. However, there is a need it to be accurate
        since we use the likelihood as a stopping criterion. That is why an average is computed
        for smoothness.
        '''
        # Append the last log likelihood computed
        self.last_likelihoods.append(self.log_like)
        # If enough likelihoods to build the mean, remove the oldest one.
        if len(self.last_likelihoods) > self.nb_average_likelihood:
            del(self.last_likelihoods[0])
        # Take the mean of the last likelihoods
        self.average_log_like = np.mean(np.array(self.last_likelihoods))
        # Store the average likelihood
        self.log_likelihood_list.append(self.average_log_like)

    def average_params(self):
        '''Averages the parameters in order to smooth the variance.
        Will take, for example, the last self.nb_average_param betas computed to make
        a better approximation of beta. Same for C. This method adds the last parameters
        computed to update the mean of the parameter. If there is not enough betas or
        Sigmas (less than self.nb_average_param), the mean will be computed on those parameters.
        Args :
            None
        Returns :
            None but update the mean of the last self.nb_average_param parameters.
        '''

        # Remove the oldest parameters and add the more recent one.
        self.last_betas[1:self.nb_average_param] = torch.clone(
            self.last_betas[0: self.nb_average_param - 1].detach())
        self.last_betas[0] = torch.clone(self.beta.detach())
        self.last_Cs[1:self.nb_average_param] = torch.clone(
            self.last_Cs[0: self.nb_average_param - 1].detach())
        self.last_Cs[0] = torch.clone(self.C.detach())
        # Update the mean of the parameter

        # If have enough parameters
        if self.iteration_cmpt > self.nb_average_param:
            self.C_mean = torch.mean(self.last_Cs, axis=0)
            self.beta_mean = torch.mean(self.last_betas, axis=0)

        # If not enough parameters.
        else:
            # to keep track of the number of parameters computed for the mean
            self.iteration_cmpt += 1
            self.C_mean = torch.sum(self.last_Cs, axis=0) / self.iteration_cmpt
            self.beta_mean = torch.sum(
                self.last_betas, axis=0) / self.iteration_cmpt

    def fit(self, Y, O, covariates,  N_iter_max=500, lr=0.1,optimizer=torch.optim.Adagrad,
             VR='SAGA', batch_size=40,acc=0.005, 
             nb_plateau=15, nb_trigger=5,good_init=True, verbose=False, ):
        '''Batch gradient ascent on the log likelihood given the data. Infer
        p_theta with importance sampling and then computes the gradients by hand.
        At each iteration, look for the right importance sampling law running
        find_batch_mode. Given this mode, estimate the variance and  compute
        the weights required to estimate p_theta. Then, derive the gradients.
        Note that it only needs to know the weights to get gradients. The mean of the
        weights gives the estimated likelihood that is used as stopping criterion for
        the algorithm.

        Args :
               Y: pd.DataFrame of size (n, p). The counts
               O: pd.DataFrame of size (n,p). The offset
               covariates: pd.DataFrame of size (n,p).
               N_iter_max: int, optional. The maximum number of times the algorithm will loop over
                   the data. Will see N_epoch times each sample. Default is 500.
               lr: float greater than 0, optional. The learning rate of the batch gradient ascent.
                   Default is 0.1.
               optimizer : torch.optim.optimizer object, optional. The optimizer used.
               VR : string, optional. the Variance Reductor we want to use. Should be one of those :
                   - 'SAGA'
                   - 'SAG'
                   - 'SVRG'
                   - None
                   If None, we are not doing any variance reduction. Else the variance of
                   the gradient will be reduced using one of the three methods.
               batch_size : int between 2 and n (included), optional. The batch size of the batch
                   gradient ascent. Default is 40.
                   Default is torch.optim.Adagrad.
               acc: float strictly between 0 and 1, optional. The accuracy when computing
                   the estimation of p_theta. The lower the more accurate but
                   the slower the algorithm. Will sample int(1/acc) gaussians
                   to estimate the likelihood. Default is 0.005.
               nb_plateau : int, optional. The criterion you want to use. The algorithm
                   will stop if the criterion go past nb_plateau. Default is 15.
               nb_trigger : int, optional. The criterion will increase if the average likelihood
                   has not increased in nb_trigger epoch. Default is 5.
               good_init: Bool, optional. If True, will do an initialization that is not random.
                   Takes some time. Default is True.
               verbose: Bool, optional. If True, will plot the evolution of the algorithm 
                   in real time. Default is False. 
       Returns:
           None, but updates the parameter beta and C. Note that beta_mean
           and C_mean are more accurate and achieve a better likelihood in general.
        '''
        N_iter_max_mode=100
        lr_mode=0.3
        self.t0 = time.time()  # To keep track of the time
        self.nb_plateau = nb_plateau
        self.nb_trigger = nb_trigger
        self.batch_size = batch_size
        # Will sample 1/acc gaussians to estimate the likelihood.
        self.N_samples = int(1 / acc)
        self.init_data(Y, O, covariates, good_init)  # Initialize the data.
        # Optimizer on C and beta
        self.optimizer = optimizer([self.beta, self.C], lr=lr)
        self.mode_step_sizes += lr_mode
        # initialize the Variance Reductor.
        if VR == 'SAGA':
            vr = SAGARAD([self.beta, self.C], self.n)
        elif VR == 'SAG':
            vr = SAGRAD([self.beta, self.C], self.n)
        elif VR == 'SVRG':
            vr = SVRGRAD([self.beta, self.C], self.n)
        else:
            vr = None
        # to keep track of the best log_likelihood.
        self.max_log_like = -100000000
        for j in tqdm(range(N_iter_max)):
            # init the log likelihood. Will add each log likelihood of
            # each batch to get the log likelihood of the whole dataset.
            log_like = 0
            self.j = j
            for Y_b, covariates_b, O_b, selected_indices in self.get_batch(
                    batch_size):
                self.optimizer.zero_grad()  # zero_grad the optimizer
                # Store the batches for a nicer implementation.
                self.Y_b, self.covariates_b, self.O_b = Y_b.to(
                    device), covariates_b.to(device), O_b.to(device)
                self.selected_indices = selected_indices
                # compute the log likelihood of the batch and add it to log_likelihood
                # of the whole dataset.
                # Note that there is a need to call this function in order to be
                # able to call self.get_batch_grad_(C/beta)()
                log_like += self.infer_batch_p_theta(
                    N_iter_max_mode, lr_mode).item()
                # add a minus since pytorch minimizes a function.
                batch_grad_C = -self.get_batch_grad_C()
                batch_grad_beta = -self.get_batch_grad_beta()
                self.t_grad_estim_list.append(time.time() - self.t_grad_estim)
                # Given the gradients of the batch, update the variance
                # reducted gradients if needed.
                # Note that there is a need to give the gradient of each sample in the
                # batch, not the average gradient of the batch.
                if vr is not None:
                    vr.update_new_grad(
                        [batch_grad_beta, batch_grad_C], selected_indices)
                else:
                    self.beta.grad = torch.mean(batch_grad_beta, axis=0)
                    self.C.grad = torch.mean(batch_grad_C, axis=0)
                # optimize beta and C given the gradients.
                self.optimizer.step()
                self.average_params()  # keep track of some stat
            self.running_times.append(time.time() - self.t0)
            # The log likelihood of the whole dataset.
            self.log_like = log_like / self.n * batch_size
            # Average the  log likelihood for a criterion less random.
            self.average_likelihood()
            crit = self.compute_criterion(verbose)  # compute the criterion
            # (i.e., check if the likelihood has increased)
            # If the likelihood has not improved for self.nb_plateau iteration,
            # the algorithm stopped.
            if crit > self.nb_plateau - 1:
                print('Algorithm stopped after ', j, ' iterations')
                self.fitted = True
                break
            # if we use SVRG VR, we need to compute the gradient of the
            # whole dataset to update the average gradient.
            if VR == 'SVRG':
                self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
                self.selected_indices = np.arange(0, self.n)
                self.get_gradient_requirement(N_iter_max_mode, lr_mode)
                full_grad_C = -self.get_batch_grad_C()
                full_grad_beta = -self.get_batch_grad_beta()
                vr.update_table([full_grad_beta, full_grad_C])
            # Next, we lower the learning rate (that finds the mode) since after one epoch,
            # we are not very far from the new mode. Indeed, we found the mode for previous
            # beta and C, and we now need to find the new mode for the new beta and C. We
            # will begin our ascent from the previous mode found. However, since beta and
            # C won't move very much, the mode for the new beta and C won't move very much
            # either. We can lower the learning rate so that the optimizer will only adjust
            # a little bit its starting position.

            if j == 0:
                lr_mode /= 10
            if j == 1 or j == 2:
                lr_mode /= 2
        # The model has been fitted.
        self.fitted = True

    def compute_best_loglike(
            self, acc=0.001, N_iter_max_mode=300, lr_mode=0.001):
        '''Estimate the best likelihood of the model, i.e. the likelihood
        estimated with beta_mean and C_mean.

        Args:
            acc: float greater than 0, optional. The accuracy wanted for
                the estimation of the likelihood. Default is 0.001.
            N_iter_max_mode : int, optional. The maximum number of iteration
                to do to find the mode. Default is 300.
            lr_mode : float greater than 0, optional. The learning of the gradient ascent
                finding the mode. Default is 0.001.
        Returns :
            float (non positive). The estimated log likelihood of beta_mean and C_mean
        '''
        self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
        self.selected_indices = np.arange(0, self.n)
        self.N_samples = int(1 / acc)
        # Set beta and C as beta_mean and C_mean to compute the likelihood.
        self.beta = torch.clone(self.beta_mean)
        self.C = torch.clone(self.C_mean)
        # Infer p_theta
        self.best_log_like = self.infer_batch_p_theta(N_iter_max_mode, lr_mode)
        return self.best_log_like

    def compute_criterion(self, verbose=True):
        '''Updates the criterion of the model. The criterion counts the
        number of times the likelihood has not improved. We also append
        the criterion in a list in order to plot it after.
        Args :
            verbose: bool. If True, will print the criterion whenever it increases.
                Default is True.
        Returns : int. The criterion.
        '''
        if verbose:
            # Print the average log likelihood as indicator
            print('Average log likelihood : ', self.average_log_like)
        # If the average likelihood has increased,
        # update the maximum likelihood found
        if self.average_log_like > self.max_log_like:
            self.counter_list.append(self.counter_list[-1])
            self.max_log_like = self.average_log_like
        # Else add one to the counter
        else:
            self.counter_list.append(self.counter_list[-1] + 1)
        triggered = int(self.counter_list[-1]
                        - self.counter_list[-self.nb_trigger - 1] > self.nb_trigger - 1)
        self.crit_list.append(self.crit_list[-1] + triggered)
        if triggered > 0 and verbose:
            print(' Criterion updated : ',
                  self.crit_list[-1], '/', self.nb_plateau)
        return self.crit_list[-1]

    def infer_batch_p_theta(self, N_iter_max_mode, lr_mode):
        '''Infer p_theta that is computed for a batch of the dataset. The
        parameter Y,O,cov are in the object itself, so that there is no need
        to pass them in argument.
        Args :
            N_iter_max_mode : int. The maximum number of iteration
                to do to find the mode of the posterior.
            lr_mode: postive float. The learning rate of the gradient
                ascent finding the mode.
        '''
        # Get the gradient requirement. It also computes the weights of the IMPS
        # that are stored in the object.
        self.get_gradient_requirement(N_iter_max_mode, lr_mode)
        # Take the log of the weights and adjust with the missing constant
        # self.const that has been removed before to avoid numerical 0.
        log = torch.log(torch.mean(self.weights, axis=0)) + self.const
        # Return the mean of the log likelihood of the batch
        return torch.mean(log)

    def get_gradient_requirement(self, N_iter_max_mode, lr_mode):
        '''Does all the operation needed to compute the gradients.
        The requirement are the gaussian samples and the weights, which are
        computed here. The gaussians samples needs to be sampled from the
        right mean and variance, found by calling find_batch_mode and
        get_batch_best_var methods. The formula of the variance
        can be found in the mathematical description.
        Args:
            N_iter_mode : int. The maximum number of iterations to do
                to find the mode.
            lr_mode : float greater than 0. The learning rate of the
                gradient ascent finding the mode.
        Returns:
            None but computes the weights stored in the object.
        '''
        # get the mode
        self.t_mode = time.time()
        self.find_batch_mode(N_iter_max_mode, lr_mode)
        # Thanks to the mode, the best variance can be computed.
        self.get_batch_best_var()
        self.t_mode_list.append(time.time() - self.t_mode)
        self.t_grad_estim = time.time()
        # get the samples generated with the mean (mode) and variance found.
        self.samples = sample_gaussians(
            self.N_samples, self.batch_mode, self.sqrt_Sigma_b)
        # get the weights
        self.weights = self.get_batch_weights()

    def get_batch_weights(self):
        '''Compute the weights of the IMPS formula. Given the gaussian samples
        stored in the object,the weights are computed as the ratio of the
        likelihood of the posterior and the likelihood of the gaussian samples.
        Note that it first compute the logarithm of the likelihood of the posterior
        and the logarithm of the gaussian samples, then remove the maximum of
        the difference to avoid numerical zero, and takes the exponential.
        We keep in memory the constant removed to get it back later.

        Args: None

        Returns: torch.tensor of size (N_samples,N_batch). The computed weights.
        '''
        # Log likelihood of the posterior
        self.log_f = self.batch_un_log_posterior(self.samples)
        # Log likelihood of the gaussian density
        self.log_g = log_gaussian_density(
            self.samples, self.batch_mode, self.Sigma_b)
        # Difference between the two logarithm
        diff_log = self.log_f - self.log_g
        self.const = torch.max(diff_log, axis=0)[0]
        # remove the maximum to avoid numerical zero.
        diff_log -= torch.max(diff_log, axis=0)[0]
        weights = torch.exp(diff_log)
        return weights

    def get_batch_best_var(self):
        '''Compute the best variance for the importance law. Given the mode,
        derive the best variance that fits the posterior. Please check the
        mathematical description of the package to find out why those
        computations are made.

        Args: None

        Returns: None but compute the best covariance matrix and
            its square root, stored in the IMPS_PLN object.
        '''
        batch_matrix = torch.matmul(
            self.C.unsqueeze(2),
            self.C.unsqueeze(1)).unsqueeze(0)
        CW = torch.matmul(
            self.C.unsqueeze(0),
            self.batch_mode.unsqueeze(2)).squeeze()
        common = torch.exp(
            self.O_b
            + self.covariates_b @ self.beta
            + CW
        ).unsqueeze(2).unsqueeze(3)
        prod = batch_matrix * common
        # The hessian of the posterior
        Hess_post = torch.sum(prod, axis=1) + torch.eye(self.q).to(device)
        self.Sigma_b = torch.inverse(Hess_post.detach())
        # Add a term to avoid non-invertible matrix.
        eps = torch.diag(torch.full((self.q, 1), 1e-8).squeeze()).to(device)
        self.sqrt_Sigma_b = TLA.cholesky(self.Sigma_b + eps)

    def get_batch_grad_beta(self):
        ''' Computes the gradient with respect to beta of the log likelihood
        for the batch. The derivation of the formula is in the README.

        Args: None

        Returns: torch.tensor of size (batch_size,d,p). The gradient wrt beta.
        '''
        first = torch.matmul(
            self.covariates_b.unsqueeze(2),
            self.Y_b.unsqueeze(1).double())
        XB = torch.matmul(self.covariates_b.unsqueeze(1),
                          self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(self.C.reshape(1, 1, self.p, 1, self.q),
                          self.samples.unsqueeze(2).unsqueeze(4)).squeeze()
        Xexp = torch.matmul(self.covariates_b.unsqueeze(0).unsqueeze(3),
                            torch.exp(self.O_b + XB + CV).unsqueeze(2))
        WXexp = torch.sum(
            torch.multiply(
                self.weights.unsqueeze(2).unsqueeze(3),
                Xexp), axis=0)
        sec = WXexp / (torch.sum(self.weights,
                                 axis=0).unsqueeze(1).unsqueeze(2))
        return first - sec

    def get_batch_grad_C(self):
        '''Computes the gradient with respect to C of the log likelihood for
        the batch. The derivation of the formula is in the README.

        Args: None

        Returns: torch.tensor of size (batch_size,d,p). The gradient wrt C.
        '''
        XB = torch.matmul(
            self.covariates_b.unsqueeze(1),
            self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(
            self.C.reshape(1, 1, self.p, 1, self.q),
            self.samples.unsqueeze(2).unsqueeze(4)
        ).squeeze()
        Ymoinsexp = self.Y_b - torch.exp(self.O_b + XB + CV)
        outer = torch.matmul(Ymoinsexp.unsqueeze(3), self.samples.unsqueeze(2))
        denum = torch.sum(self.weights, axis=0)
        num = torch.multiply(self.weights.unsqueeze(2).unsqueeze(3), outer)
        batch_grad = torch.sum(
            num / (denum.unsqueeze(0).unsqueeze(2).unsqueeze(3)), axis=0)
        return batch_grad

    def show_Sigma(self):
        '''Displays Sigma'''
        sns.heatmap(self.get_Sigma())
        plt.show()

    def __str__(self):
        '''Show the model, Sigma and the likelihood.'''
        self.best_log_like = max(self.log_likelihood_list)
        self.show()
        self.show_Sigma()
        return ''

    def show(self, save=False, name_doss='IMPS_PLN_graphic'):
        """Show some useful stats of the model. Plot the estimated log_likelihood
        and the criterion in the y axis with the runtime in the x-axis. The model
        should have been fitted beofre calling show().

        Args :
            save: bool, optional. If True, the graphic will be saved. If false, won't be saved.
                default is False.
            name_doss: str, optional. The name of the file the graphic will be saved to.
                Default is 'IMPS_PLN_graphic'.
        Returns :
                None but displays the figure. It can also save the figure if save = True.
        Raises:
            AttributeError when the model has not been fitted.
        """
        # Make sure the model has been fitted
        if not self.fitted:
            print(
                'Please fit the model before by calling model.fit(Y,O,covariates,N_epoch,acc)')
            raise AttributeError
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            plt.subplots_adjust(hspace=0.4)
            abscisse = self.running_times

            print(
                'Max likelihood:', np.max(
                    np.array(
                        self.log_likelihood_list)))
            # plot the negative likelihood of the model
            ax[0].plot(np.arange(0, len(self.log_likelihood_list)),
                       -np.array(self.log_likelihood_list))
            ax[0].set_title('Smoothed negative log likelihood')
            ax[0].set_ylabel('Negative loglikelihood')
            ax[0].set_xlabel('Seconds')
            ax[0].set_yscale('log')
            # Plot the criteria of the model
            ax[1].plot(np.arange(0, len(self.crit_list)), self.crit_list)
            ax[1].set_title('Number of epoch the likelihood has not improved')
            ax[1].set_xlabel('Seconds')
            # save the graphic if needed
            if save:
                plt.savefig(name_doss)
            plt.show()
            if self.p >400: 
                print('The heatmap only displays Sigma[:400,:400]') 
                sns.heatmap(self.get_Sigma()[:400,:400].cpu().detach())
            else : 
                sns.heatmap(self.get_Sigma().cpu().detach())
    def get_beta(self):
        '''Getter for beta. Returns the mean of the last betas computed to reduce variance.'''
        return self.beta_mean.detach()

    def get_Sigma(self):
        '''Getter. Get Sigma by computing C_mean@C_mean^T. We take C_mean to reduce variance. '''
        return (self.C_mean.detach()) @ (self.C_mean.detach().T)

    def get_C(self):
        '''Getter for C. We return C_mean to reduce variance'''
        return self.C_mean.detach()

    def find_batch_mode(self, N_iter_max, lr, eps=9e-3):
        '''Find the mode of the posterior with a gradient ascent.
        The last mode computed is used as starting point. However,
        each mode depends on the batch (Y_b,O_b, covariates_b), so that
        there is a need to know from which indices we have selected the batch.

        Args:
            N_iter_max: int. The maximum number of iteration to do
                to find the mode.
            lr: positive float. The learning rate of the optimizer for the
                gradient ascent.
            eps: positive float, optional. The tolerance. The algorithm will
                stop if the maximum of |W_t-W_{t-1}| is lower than eps, where W_t
                is the t-th iteration of the algorithm.This parameter changes a lot
                the resulting time of the algorithm. Default is 9e-3.

        Returns :
            None, but compute and stock the mode in self.batch_mode and the starting point.
        '''
        # The loss used use for the gradient ascent.
        def batch_un_log_posterior(W):
            return batch_log_P_WgivenY(
                self.Y_b, self.O_b, self.covariates_b, W, self.C, self.beta)
        self.batch_un_log_posterior = batch_un_log_posterior
        # Get the corresponding starting point.
        W = self.starting_point[self.selected_indices]
        W.requires_grad = True
        # If we have seen enough data, we set the learning rate to zero
        # since we will actually use the previous learning rate.
        if self.j > 5:
            lr = 0
        optim = torch.optim.Rprop([W], lr=lr)
        criterion = 2 * eps
        old_W = torch.clone(W)
        i = 0
        stop_condition = False
        while i < N_iter_max and stop_condition == False:
            # When self.j >5, will move just a little bit from the previous C and beta.
            # Thus, the mode won't move very much. We take the previous step_sizes of the optimizer
            # in order to reach the maximum faster.
            if i == 1 and self.j > 5:
                optim.state_dict()['state'][0]['step_size'] = 5 * \
                    self.mode_step_sizes[self.selected_indices, :]
            loss = -torch.mean(self.batch_un_log_posterior(W))
            # Propagate the gradients
            loss.backward()
            # Update the parameter
            optim.step()
            crit = torch.max(torch.abs(W - old_W))
            optim.zero_grad()
            if crit < eps and i > 2:  # we want to do at least 3 iteration per loop.
                stop_condition = True
            old_W = torch.clone(W)
            i += 1
        # Keep the number of iteration as information
        self.nb_iteration_list.append(i)
        # Stock the mode
        self.batch_mode = torch.clone(W.detach())
        # Stock the starting point for the next epoch.
        self.starting_point[self.selected_indices] = torch.clone(W.detach())
        self.mode_step_sizes[self.selected_indices,
                             :] = optim.state_dict()['state'][0]['step_size']

    def plot_runtime(self):
        '''
        Shows different runtimes of the .fit() method. It shows what computation takes
        time. Do so to estimate what should be lowered or increased. For example,
        if the Gradient estimation running time is very low compared to the time took to find the mode,
        then the overall running time won't increase very much if the accuracy parameter is lowered.
        Args:
            None
        Returns:
            None but displays a figure.
        '''
        l_estim = imps.t_grad_estim_list
        l_mode = imps.t_mode_list
        l_total = np.array(l_estim) + np.array(l_mode)

        plt.plot(np.arange(len(l_estim)), l_estim, label='Gradient estimation')
        plt.plot(np.arange(len(l_mode)), l_mode, label='Finding mode')
        plt.plot(np.arange(len(l_mode)), l_total, label='Total runtime')
        plt.ylabel('Running time')
        plt.xlabel('Iteration')
        plt.legend()
        print('Total time :', np.sum(l_total))
        plt.show()

#################################### FastPLN object ######################


def ELBO(Y, O, covariates, M, S, Sigma, beta):
    '''Compute the ELBO (Evidence LOwer Bound. See the doc for more details
    on the computation.

    Args:
        Y: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        Sigma: torch.tensor. Model parameter with size (p,p)
        beta: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient. The ELBO.
    '''
    n, p = Y.shape
    SrondS = torch.multiply(S, S)
    OplusM = O + M
    MmoinsXB = M - torch.mm(covariates, beta)
    tmp = torch.sum(torch.multiply(Y, OplusM)
                    - torch.exp(OplusM + SrondS / 2)
                    + 1 / 2 * torch.log(SrondS)
                    )
    DplusMmoinsXB2 = torch.diag(
        torch.sum(SrondS, dim=0)) + torch.mm(MmoinsXB.T, MmoinsXB)
    
    
    tmp -= 1 / 2 * torch.trace(
        torch.mm(
            torch.inverse(Sigma),
            DplusMmoinsXB2
        )
    )
    tmp -= n / 2 * torch.logdet(Sigma)
    tmp -= torch.sum(log_stirling(Y))
    tmp += n * p / 2 
    return tmp


class fastPLN():
    '''Implement the variational algorithm infering the parameters of the PLN model,
    with a closed form for the M step and a gradient step for the VE step. Any value of n
    and p can be taken.
    '''

    def __init__(self):
        '''Defines some usefuls lists and variables for the object. A deeper initalization is done
        in the init_data() method, once the dataset is available.
        '''
        self.window = 3
        self.fitted = False

    def init_data(self, Y, O, covariates, good_init):
        '''Initialize the parameters with the right shape given the data.

        Args:
              Y: pd.DataFrame of size (n, p). The counts
              O: pd.DataFrame of size (n,p). The offset
              covariates: pd.DataFrame of size (n,p)
              good_init: bool. If True,  a good initialization (not random)
                  will be performed. Takes some time.
        Returns:
            None but initialize some useful data.
        '''
        # Known variables
        if self.fitted == False:
            # import the data. We take either pandas.DataFrames or torch.tensor
            # pandas.DataFrame
            try: 
                self.Y = torch.from_numpy(Y.values).to(device)
                self.O = torch.from_numpy(O.values).to(device)
                self.covariates = torch.from_numpy(covariates.values).to(device)
            # torch.tensor (if not torch.tensor, will launch an error after) 
            except: 
                try:   
                    self.Y = Y.to(device)
                    self.O = O.to(device)
                    self.covariates = covariates.to(device)
                except: 
                    raise ValueError('Each of Y,O, covariates should be either a pandas.DataFrame or a torch.tensor')
        

        if self.fitted == False:
            try: 
                self.Y = torch.from_numpy(Y.values).to(device)
                self.O = torch.from_numpy(O.values).to(device)
                self.covariates = torch.from_numpy(covariates.values).to(device)
            # torch.tensor (if not torch.tensor, will launch an error after) 
            except: 
                try:   
                    self.Y = Y.to(device)
                    self.O = O.to(device)
                    self.covariates = covariates.to(device)
                except: 
                    raise ValueError('Each of Y,O, covariates should be either a pandas.DataFrame or a torch.tensor')
        
            self.n, self.p = self.Y.shape
            self.d = self.covariates.shape[1]

            # Lists to store some stats
            self.running_times = list()
            self.deltas = [1] * self.window
            self.normalized_ELBOs = list()
            print('Initialization ...')
            if good_init:
                # Model parameters
                poiss_reg = Poisson_reg()
                poiss_reg.fit(self.Y, self.O, self.covariates)
                self.beta = torch.clone(poiss_reg.beta.detach()).to(device)
                self.Sigma = init_Sigma(
                    self.Y, self.O, self.covariates, self.beta).to(device)
                # Initialize C in order to initialize M.
                self.C = TLA.cholesky(self.Sigma).to(device)
                # Variational parameter
                self.M = init_M(
                    self.Y,
                    self.O,
                    self.covariates,
                    self.beta,
                    self.C,
                    300,
                    0.1).to(device) 
                self.M.requires_grad_(True)
                
            else:
                # Random initialization with the right shape
                self.beta = torch.randn((self.d, self.p), device = device)
                self.Sigma = torch.diag(torch.ones(self.p)).to(device)
                self.M = torch.ones(self.n, self.p, device = device)
            print('Initialization finished')

            # No better initialization possible for S
            self.S = 1 / 2 * torch.ones((self.n, self.p)).to(device)

    def compute_ELBO(self):
        '''Compute the ELBO with the parameter of the model.'''
        return ELBO(self.Y, self.O, self.covariates,
                    self.M, self.S, self.Sigma, self.beta)

    def fit(self, Y, O, covariates, N_iter_max=200, lr=0.1,optimizer=torch.optim.Rprop, 
            tol=1e-1,good_init=True, verbose=False):
              
        '''Main function of the class. Infer the best parameter Sigma and beta given the data.

        Args:
            Y: pd.DataFrame of size (n, p). The counts.
            O: pd.DataFrame of size (n,p). The offset
            covariates: pd.DataFrame of size (n,p)
            N_iter_max: int, optional. The maximum number of iteration.
                Default is 200.
            lr: positive float, optional. The learning rate of the optimizer. Default is 0.1.
            optimizer: objects that inherits from torch.optim, optional. The optimizer wanted.
                Default is torch.optim.Rprop.
            tol: non negative float, optional. The algorithm will stop if the ELBO has not
                improved more than tol. Default is 1e-1.
            good_init: bool, optional. If True, will do a smart initialization instead of
                a random initialization. Default is True.
            verbose: bool, optional. If True, will print some stats during the fitting. Default is
                False.

        Returns:
            None but update the parameter C and beta of the object.
        '''
        self.t0 = time.time()
        # Initialize the data
        self.init_data(Y, O, covariates, good_init)
        self.optimizer = optimizer([self.S, self.M], lr=lr)
        stop_condition = False
        i = 0
        self.old_beta = torch.clone(self.beta.detach())
        self.old_Sigma = torch.clone(self.Sigma.detach())
        delta = 2 * tol
        while i < N_iter_max and stop_condition == False:
            # VE step
            self.optimizer.zero_grad()
            self.M.grad = -self.grad_M()
            self.S.grad = -self.grad_S()
            self.optimizer.step()
            # M step
            self.beta = self.closed_beta()
            self.Sigma = self.closed_Sigma()
            # Keep records
            self.normalized_ELBOs.append(
                -1 / self.n * self.compute_ELBO().item())
            self.running_times.append(time.time() - self.t0)
            # Criterions
            if i > self.window - 1:  # To be sure we have seen enough data
                delta = abs(
                    self.normalized_ELBOs[-1] - self.normalized_ELBOs[-1 - self.window])
                self.deltas.append(delta)
            # Condition to see if the tolerance has been reached.
            if delta < tol:
                stop_condition = True
            # Print some stats if wanted.
            if i % 10 == 0 and verbose:
                print('Iteration number: ', i)
                print('-------UPDATE-------')
                print('Delta : ', delta)
            i += 1
        if stop_condition:
            print('Last delta: {},  reached in {} iterations'.format(delta, i))
        else:
            print('Maximum number of iterations reached : ',
                  N_iter_max, 'last delta = ', delta)
        self.fitted = True

    def grad_M(self):
        '''Compute the gradient of the ELBO with respect to M'''
        grad = self.Y - torch.exp(self.O + self.M +
                                  torch.multiply(self.S, self.S) / 2)
        grad -= torch.mm(self.M - torch.mm(self.covariates,
                         self.beta), torch.inverse(self.Sigma))
        return grad

    def grad_S(self):
        '''Compute the gradient of the ELBO with respect to S'''
        grad = torch.div(1, self.S)
        grad -= torch.multiply(self.S, torch.exp(self.O +
                               self.M + torch.multiply(self.S, self.S) / 2))
        grad -= torch.mm(self.S,
                         torch.diag(torch.diag(torch.inverse(self.Sigma))))
        return grad

    def closed_Sigma(self):
        '''Closed form for Sigma for the M step.'''
        n, p = self.M.shape
        MmoinsXB = self.M - torch.mm(self.covariates, self.beta)
        closed = torch.mm(MmoinsXB.T, MmoinsXB)
        closed += torch.diag(torch.sum(torch.multiply(self.S, self.S), dim=0))
        return 1 / (n) * closed

    def closed_beta(self):
        '''Closed form for beta for the M step.'''
        return torch.mm(
            torch.mm(
                torch.inverse(torch.mm(
                    self.covariates.T,
                    self.covariates)),
                self.covariates.T),
            self.M)

    def show_Sigma(self):
        '''Simple method that displays Sigma to see the global structure.'''
        sns.heatmap(self.Sigma.detach().cpu().numpy())
        plt.show()

    def show(self, name_doss='fastPLN_graphic', save=False):
        '''displays some useful stats of the model.

        Args :
            'name_doss' : str. The name of the file the graphic will be saved to.
                Default is 'fastPLN_graphic'.
            'save' : bool. If True, the graphic will be saved. If false, won't be saved.
                Default is False.

        Returns :
                None but displays the figure. It can also save the figure if save = True.
        '''
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        abscisse = self.running_times
        plt.subplots_adjust(hspace=0.4)

        ax[0].plot(abscisse, np.array(self.normalized_ELBOs),
                   label='Negative ELBO')
        ax[0].set_title('Negative ELBO')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Seconds')
        ax[0].set_ylabel('ELBO')
        ax[0].legend()

        ax[1].plot(abscisse[self.window:],
                   self.deltas[self.window:], label='Delta')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Seconds')
        ax[1].set_ylabel('Delta')
        ax[1].set_title('Increments')
        ax[1].legend()
        if save:
            plt.savefig(name_doss)
        plt.show()
        if self.p >400: 
            print('The heatmap only displays Sigma[:400,:400]') 
            sns.heatmap(self.get_Sigma()[:400,:400].cpu().detach())
        else : 
            sns.heatmap(self.get_Sigma().cpu().detach())
        plt.show()
    def __str__(self):
        '''Show the stats of the model and Sigma'''
        print('Last ELBO :', -self.normalized_ELBOs[-1])
        self.show()
        return ''

    def get_Sigma(self):
        '''Getter for Sigma'''
        return self.Sigma.detach()

    def get_beta(self):
        '''Getter for beta'''
        return self.beta.detach()

    def get_C(self, q=None):
        '''Getter for C. We do an ACP on Sigma and take the q largest eigenvectors.
        if q is None, we will take p.'''
        if q is None:
            q = self.p
        return C_from_Sigma(self.Sigma, q).detach()

######################################### fastPLNPCA object ##############



def ELBO_PCA(Y, O, covariates, M, S, C, beta):
    '''compute the ELBO with a PCA parametrization'''
    n = Y.shape[0]
    q = C.shape[1]
    # Store some variables that will need to be computed twice
    A = O + torch.mm(covariates, beta) + torch.mm(M, C.T)
    SrondS = torch.multiply(S, S)
    # Next, we add the four terms of the ELBO_PCA
    YA = torch.sum(torch.multiply(Y, A))
    moinsexpAplusSrondSCCT = torch.sum(-torch.exp(A + 1 / 2 *
                       torch.mm(SrondS, torch.multiply(C, C).T)))
    moinslogSrondS = 1 / 2 * torch.sum(torch.log(SrondS))
    MMplusSrondS = torch.sum(-1 / 2 * (torch.multiply(M, M) + torch.multiply(S, S)))
    log_stirlingY = torch.sum(log_stirling(Y))
    return YA + moinsexpAplusSrondSCCT + moinslogSrondS + MMplusSrondS - log_stirlingY + n * q / 2


class fastPLNPCA():
    def __init__(self, q):
        '''Define some usefuls lists and variables for the object. A deeper initalization is done
        in the init_data() method, once the dataset is available.
        Args:
            q: int. The dimension of the latent space.
        Returns:
            A fastPLNPCA object
        '''
        self.old_loss = 1
        self.q = q
        # Lists to store some stats
        self.fitted = False
        self.window = 3

    def init_data(self, Y, O, covariates, good_init):
        '''Initialize the parameters with the right shape given the data.

        Args:
            Y: pd.DataFrame of size (n, p) or torch.tensor of size (n,p). The counts
            O: pd.DataFrame of size (n,p), or torch.tensor of size (n,p). the offset
            covariates: pd.DataFrame of size (n,p) or torch.tensor of size (n,p). The covariates
            good_init: bool. If True a good initialization (not random)
                will be performed. Takes some time.
        Returns:
            None but initialize some useful data.'''
        # Lists to store some stats.
        self.ELBO_list = list()
        self.running_times = list()
        self.deltas = [1] * self.window
        self.normalized_ELBOs = list()
        if self.fitted == False:
            # import the data. We take either pandas.DataFrames or torch.tensor
            # pandas.DataFrame
            try: 
                self.Y = torch.from_numpy(Y.values).to(device)
                self.O = torch.from_numpy(O.values).to(device)
                self.covariates = torch.from_numpy(covariates.values).to(device)
            # torch.tensor (if not torch.tensor, will launch an error after) 
            except: 
                try:   
                    self.Y = Y.to(device)
                    self.O = O.to(device)
                    self.covariates = covariates.to(device)
                except: 
                    raise ValueError('Each of Y,O, covariates should be either a pandas.DataFrame or a torch.tensor')
            self.n, self.p = self.Y.shape
            self.d = self.covariates.shape[1]
            print('Initialization ...')
            # If a good initialization is wanted.
            if self.p > 1500: 
                print('p is too large (>1500) to do a good initialization, random intialization is performed instead')
                good_init = False
            if good_init:
                poiss_reg = Poisson_reg()
                poiss_reg.fit(self.Y, self.O, self.covariates)
                # Model parameter
                self.beta = torch.clone(poiss_reg.beta.detach()).to(device)
                self.C = init_C(
                    self.Y,
                    self.O,
                    self.covariates,
                    self.beta,
                    self.q).to(device)
                # Variational parameter
                self.M = init_M(
                    self.Y,
                    self.O,
                    self.covariates,
                    self.beta,
                    self.C,
                    300,
                    0.1)
            # Else, random initalization. Faster but worst.
            else:
                self.C = torch.randn((self.p, self.q)).to(device)
                self.beta = torch.randn((self.d, self.p)).to(device)
                self.M = torch.randn((self.n, self.q)).to(device)
            print('Initialization finished')
            # Can't do any further initialization
            self.S = torch.randn((self.n, self.q)).to(device)
            # Set some gradients for optimization
            self.beta.requires_grad_(True)
            self.M.requires_grad_(True)
            self.S.requires_grad_(True)
            self.C.requires_grad_(True)

    def compute_ELBO_PCA(self):
        '''Compute the ELBO of the PCA parametrisation with the parameter of the model.'''
        return ELBO_PCA(self.Y, self.O, self.covariates,
                        self.M, self.S, self.C, self.beta)

    def fit(self, Y, O, covariates, N_iter_max=15000, lr=0.01, optimizer=torch.optim.Rprop,
            tol=1e-3, good_init=True, verbose=False):
        '''Main function of the class. Infer the best parameter C and beta given the data.

        Args:
            Y: pd.DataFrame of size (n, p). The counts.
            O: pd.DataFrame of size (n,p). The offset.
            covariates: pd.DataFrame of size (n,p).
            N_iter_max: int, optional. The maximum number of iteration. (Default is 15000).
            lr: positive float, optional. The learning rate of the optimizer. Default is 0.01.
            optimizer: objects that inherits from torch.optim, optional. The optimize wanted.
                Default is torch.optim.Rprop.
            tol: non negative float. Criterion for the model. The algorithm will
                stop if the ELBO has not improved more than tol(Default is 1e-3).
            good_init: Bool, optional. If True, will do a good initialization. Takes some time.
                Default is True.
            verbose: Bool, optional. If True, will print some stats during the fitting. Default is
                False.
        Returns:
            None but update the parameter C and beta of the object.
        '''
        self.max_Sigma = []
        self.interval = interval
        self.t0 = time.time()
        # initialize the data
        self.init_data(Y, O, covariates, good_init)
        self.optimizer = optimizer([self.beta, self.C, self.M, self.S], lr=lr)
        stop_condition = False
        i = 0
        delta = 1
        while i < N_iter_max and stop_condition == False:
            self.optimizer.zero_grad()
            loss = -self.compute_ELBO_PCA()
            loss.backward()
            self.optimizer.step()
            # Keep records
            self.normalized_ELBOs.append(-1 / self.n * loss.item())
            # Criterion
            if i > self.window - 1:
                delta = abs(
                    self.normalized_ELBOs[-1] - self.normalized_ELBOs[-1 - self.window])
                self.deltas.append(delta)
            # Condition to see if we have reached the tolerance threshold
            if abs(delta) < tol:
                stop_condition = True
            # Print some stats if we want to
            if i % 100 == 0 and verbose:
                print('Iteration number: ', i)
                print('-------UPDATE-------')
                print('Delta : ', delta)
            # Keep track of the time
            self.running_times.append(time.time() - self.t0)
            self.max_Sigma.append(torch.max(self.get_Sigma()).item())
            if self.compare_with_likelihood ==True:
                '''
                if i%interval == 0: 
                    log_like = log_likelihood(self.Y, self.O, self.covariates, self.C, self.beta).item()
                    print('loglike :', log_like)
                    self.likelihood_list.append(log_like)
                else: 
                    self.likelihood_list.append(self.likelihood_list[-1])
                '''
            i += 1
        if stop_condition:
            print('Tolerance {} reached in {} iterations'.format(tol, i))
        else:
            print('Maximum number of iterations reached : ',
                  N_iter_max, 'last delta = ', delta)
        self.fitted = True

    def gradPCA_beta(self):
        '''Compute the gradient of the ELBO with respect to beta. Sanity check'''
        matC = self.C
        CrondC = torch.multiply(matC, matC)
        SrondS = torch.multiply(self.S, self.S)
        first_term = torch.mm(self.covariates.T, self.Y)
        second_term = -1 / 2 * torch.mm(
            self.covariates.T,
            torch.exp(self.O + torch.mm(
                self.covariates,
                self.beta
            )
                + torch.mm(self.M, matC.T) + 1 / 2 * torch.mm(SrondS, CrondC.T))
        )
        return first_term + second_term

    def gradPCA_M(self):
        '''Compute the gradient of the ELBO with respect to M. Sanity check'''
        CrondC = torch.multiply(self.C, self.C)
        SrondS = torch.multiply(self.S, self.S)
        A = self.O + torch.mm(self.covariates, self.beta) + \
            torch.mm(self.M, self.C.T)
        first = torch.mm(self.Y, self.C)
        second = -1 / 2 * \
            torch.mm(torch.exp(A + 1 / 2 * torch.mm(SrondS, CrondC.T)), self.C)
        third = -self.M
        return first + second + third

    def gradPCA_S(self):
        '''Compute the gradient of the ELBO with respect to S. Sanity check'''
        matC = self.C
        CrondC = torch.multiply(matC, matC)
        SrondS = torch.multiply(self.S, self.S)
        A = self.O + torch.mm(self.covariates, self.beta) + \
            torch.mm(self.M, matC.T)
        first = -1 / 2 * \
            torch.multiply(
                self.S,
                torch.mm(
                    torch.exp(
                        A +
                        1 /
                        2 *
                        torch.mm(
                            SrondS,
                            CrondC.T)),
                    CrondC))
        second = torch.div(1, self.S)
        third = -self.S
        return first + second + third

    def gradPCA_C(self):
        '''Compute the gradient of the ELBO with respect to C. Sanity check'''
        matC = self.C
        CrondC = torch.multiply(matC, matC)
        SrondS = torch.multiply(self.S, self.S)
        first = torch.mm(self.Y.T, self.M)
        A = self.O + torch.mm(self.covariates, self.beta) + \
            torch.mm(self.M, matC.T)
        exp = torch.exp(A + 1 / 2 * torch.mm(SrondS, CrondC.T))
        second = -1 / 2 * torch.mm(exp.T, self.M) - 1 / \
            2 * torch.multiply(matC, torch.mm(exp.T, SrondS))
        return first + second

    def show(self, name_doss='fastPLNPCA_graphic', save=False):
        '''Display some useful stats of the model.

        args :
            'name_doss': str, optional. The name of the file the graphic will be saved to.
                Default is 'fastPLNPCA_graphic'.
            'save': Bool, optional. If True, the graphic will be saved. If false, won't be saved.

        returns :
                None but displays the figure. It can also save the figure if save = True.
        '''
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        abscisse = self.running_times
        plt.subplots_adjust(hspace=0.4)
        length = len(self.running_times)
        # Plot the negative ELBO
       
        ax[0].plot(abscisse[int(length/4):], - np.array(self.normalized_ELBOs)
               [int(length/4):], label='Negative ELBO')
                   
        
        ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].set_title('Negative ELBO')
        ax[0].set_ylabel('Negative ELBO')
        ax[0].set_xlabel('Seconds')

        # Plot the criteria of the algorithm.
        ax[1].plot(abscisse[self.window+int(length/4):],
                   self.deltas[self.window+int(length/4):],
                   label='Deltas')
        ax[1].set_title('Increments')
        ax[1].set_yscale('log')
        ax[1].legend()
        ax[2].plot(np.arange(len(self.max_Sigma)), self.max_Sigma)
                                  

        if save:
            plt.savefig(name_doss)
        plt.show()
        if self.p >400: 
            print('The heatmap only displays Sigma[:400,:400]') 
            sns.heatmap(self.get_Sigma()[:400,:400].cpu().detach())
        else: 
            sns.heatmap(self.get_Sigma().cpu().detach())
        plt.show()

    def __str__(self):
        '''Show the stats of the model and Sigma'''
        print('Last ELBO :', self.normalized_ELBOs[-1])
        print('Dimension of the latent space :', self.q)
        self.show()
        return ''

    def get_Sigma(self):
        '''Return the parameter Sigma of the model, that is CC^T'''
        return (self.C @ (self.C.T)).detach()

    def get_C(self):
        '''Getter for C.'''
        return self.C.detach()

    def get_beta(self):
        '''Getter for beta'''
        return self.beta.detach()
