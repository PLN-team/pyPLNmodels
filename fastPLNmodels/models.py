#!/usr/bin/env python

"""Implement all the model related to the PLN model, such as: 
    - Variational model for PLN: fastPLN. It is very fast, but 
        can't do dimension reduction. We can only infer Sigma that
        has size (p,p). 
    - Importance Sampling based model for PLN-PCA. Relatively slow 
        compared to fastPLN. However, we don't do any approximation
        and infers the MLE. 
    - Variationel model for Zero-Inflated PLN. We are maximizing the ELBO 
        but does not get back the true parameter. 
    

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
import seaborn as sns
import torch
import torch.linalg as TLA
from tqdm import tqdm

from .utils import init_C, init_M, init_Sigma, Poisson_reg, log_stirling, batch_log_P_WgivenY, MSE
from .utils import refined_MSE, RMSE
from .VRA import SAGARAD, SAGRAD, SVRGRAD 

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')
#device = torch.device('cpu') # have to deal with this. 
    
print('device ', device)





            
def log_likelihood(Y,O,covariates, C,beta, acc = 0.002, N_iter_mode = 1000, lr_mode= 0.1): 
    """Estimate the log likelihood of C and beta given Y,O,covariates. 
    The process is a little bit complicated since we need to find 
    the mode of the posterior in order to sample the right Gaussians. 
    
    Args: 
        Y: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covarites: torch.tensor. Covariates, size (n,d)
        C: torch.tensor of size (p,q)
        beta: torch.tensor of size (d,p)
        acc: positive float, optional. The accuracy you want. Basically, 
            we will sample 1/acc gaussians to estimate the likelihood. 
            Default is 0.002.
        N_iter_mode: int, optional. The number of iteration you are ready 
            to do to find the mode of the posterior. Default is 1000. Should 
            not be lower since we need a very accurate mode. 
        lr_mode: positive float, optional. The learning rate of the 
            optimizer that finds the mode. Default is 0.1
    Returns: 
        The approximate likelihood of the whole dataset.
    """
    q = C.shape[1]
    # Initialize an IMPS_PLN model that will estimate the log likelihood.  
    model = IMPS_PLN(q)
    model.init_data(Y,O,covariates)
    model.Y_b, model.O_b, model.covariates_b = model.Y, model.O, model.covariates
    model.C_mean = C
    model.beta_mean = beta
    log_like = model.compute_best_loglike(acc, N_iter_mode, lr_mode)
    del(model)
    return log_like
 
            
def sample_gaussians(N_samples, mean,sqrt_Sigma):
    '''Sample some gaussians with the right mean and variance. 
    Be careful, we ask for the square root of Sigma, not Sigma.
    
    Args: 
         N_samples : int. the number of samples you want to sample. 
         mean : torch.tensor of size (n_batch,q) 
         sqrt_Sigma : torch.tensor or size (batch_size, q, q)
         
    Returns: 
        W: torch.tensor of size (N_samples, batch_size,q). It is a vector 
        of N_samples gaussian of dimension mean.shape. For each  1< i< N_sample,
        1<k< n_batch , W[i,k] is a gaussian with mean mean[k,:] and variance 
        sqrt_Sigma[k,:,:]@sqrt_Sigma[k,:,:].  
    '''
    q = mean.shape[1]
    W_orig = torch.randn(N_samples, 1,q,1).to(device)
    # just add the mean and multiply by the square root matrice to sample from 
    # the right distribution. 
    W = torch.matmul(sqrt_Sigma.unsqueeze(0), W_orig).squeeze() + mean.unsqueeze(0)
    return W

def log_gaussian_density(W, mu_p,Sigma_p): 
    '''Compute the log density of a gaussian W of size 
    (N_samples, n_batch, q) With mean mu_p and Sigma_p.
    
    Args : 
        W: torch.tensor of size (N_samples, batch_size, q)
        mu_p : torch.tensor : the mean from which the gaussian has been sampled.
        Sigma_p : torch.tensor. The variance from which the gaussian has been sampled. 
    Returns : 
        torch.tensor. The log of the density of W, taken along the last axis.  
    '''
    dim = W.shape[-1] # dimension q
    const = torch.sqrt((2*math.pi)**dim*torch.det(Sigma_p)) # Constant of the gaussian density
    Wmoinsmu = W-mu_p.unsqueeze(0)
    inv_Sig = torch.inverse(Sigma_p)
    # Log density of a gaussian. 
    log_d = -1/2*torch.matmul(
                            torch.matmul(inv_Sig.unsqueeze(0), 
                                         Wmoinsmu.unsqueeze(3)).squeeze().unsqueeze(2),
                            Wmoinsmu.unsqueeze(3))
    return log_d.squeeze() - torch.log(const)






class IMPS_PLN():
    '''Maximize the likelihood of the PLN-PCA model. The main method
    is the fit() method that fits the model. Most of the others 
    functions are here to support the fit method. Any value of n can be taken. 
    However, p should not be greater than 200, and q should not be greater than
    20. The greater q and p, the lower the accuracy should be taken.
    '''
    
    def __init__(self, nb_average_param = 100, nb_average_likelihood = 8):
        '''Init method. We set some global parameters of the class, such as 
        the dimension of the latent space, the number of elements we take 
        to set an average parameter that should be more accurate. We set 
        also nb_plateau, the threshold for the criterion. If 
        self.crit>nb_plateau, then the fit() method stops. nb_plateau 
        corresponds to the number of epochs we do without improving 
        the average likelihood. 
        
        Args : 
            q : int. The dimension of the latent layer you want for the PLN-PCA model. 
            nb_average_param: int, optional. We will average the parameter to get 
                parameters with lower variance. nb_average pa10:30amram tells 
                the number of parameter we take to build the mean. Should 
                not be changed since not very important. Default is 100.  
            nb_average_likelihood: int, optional. We will average the log_likelihood 
                of the model. nb_average likelihood tells the number of 
                likelihood we will take to build the mean likelihood. Should
                not be changed since not very important. Note that this 
                parameter is actually changing the algorithm (just a little bit)
                since the stopping criterion depends directly on the average likelihood.
                Default is 8. 
        Returns: 
            An IMPS_PLN object. 
        '''
        self.nb_average_likelihood = nb_average_likelihood
        self.nb_average_param = nb_average_param 
        self.running_times = list() # list to store the running times for a nice plot
        self.log_likelihood_list = list() # list to store the likelihood to plot it after
        self.last_likelihoods = list() # list that will store the last likelihoods
                                       # in order to take the mean of those likelihoods to smooth it. 
        self.criteria_list = list() # list that will store all the criterion. 
        self.nb_iteration_list = list() # list that will store the number of 
                                        # iteration we are doing at for each gradient ascent
                                        # that finds the mode. 
        self.epoch_time = list()
        
    def init_data(self, Y, O, covariates,q, good_init): 
        '''Initialise some usefuls variables given the data. 
        We also initialise C and beta. 
        
        Args : 
               Y: torch.tensor of size (n, p). The counts
               O: torch.tensor of size (n,p). the offset
               covariates: torch.tensor of size (n,p) 
        Returns: 
                None
        '''
        self.q = q # the dimension of the latent space 
        self.fitted = False # bool variable to know if we have fitted the model. 
        self.cmpt = 0 # variable that counts some iterations. 
        self.crit_cmpt = 0 
        self.crit_cmpt_list = list()
        #data 
        self.Y = Y.float().to(device)
        self.O = O.to(device)
        self.covariates = covariates.to(device)
        
        self.n = Y.shape[0] 
        self.p = Y.shape[1]
        self.d = covariates.shape[1]
        # Tensor that will store the starting point for the 
        # gradient descent that finds the mode for IMPS. 
        self.starting_point = torch.zeros(self.n, self.q, device = device, requires_grad = False) 
        self.mode_optimizer = torch.optim.Rprop([self.starting_point], lr = 0.1)
        
        # Initialization of the last beta and C. 
        self.last_betas = torch.zeros(self.nb_average_param,self.d,self.p) 
        self.last_Cs = torch.zeros(self.nb_average_param, self.p, self.q) 
        
        
        if good_init : 
            print('Intialization ...') 
            # initialization for beta with a poisson regression.
            poiss_reg = Poisson_reg()
            poiss_reg.fit(Y,O,covariates)
            self.beta = torch.clone(poiss_reg.beta.detach())#.to(device)
            # initialization for C with an array of size (p,q) taking the q vectors associated to 
            # the q largest eigenvectors of the estimated variance of log(Y).
            self.C = init_C(Y, O, covariates, self.beta, self.q)#.to(device)
            print('Initalization done')
        else : 
            self.beta = torch.randn((self.n, self.p)) 
            self.C = torch.randn((self.p, self.q))
        # mean of the last parameters, that are supposed to be more accurate.
        self.C_mean = torch.clone(self.C.detach())
        self.beta_mean = torch.clone(self.beta.detach())
        
    def get_batch(self,batch_size, save_batch_size = True): 
        '''Get the batches required to do a  minibatch gradient ascent.  
        
        Args: 
            batch_size: int. The batch size you want. Should be lower than n.  
            save_batch_size: bool, optional. If True, will set self.batch_size as batch_size. 
                Default is True. 
                
        Returns: a generator. Will generate n//batch_size + 1 batches of
            size batch_size (except the last one since the rest of the 
            division is not always 0)
        '''
        indices = np.arange(self.n)
        #shuffle the indices to avoid a regular path. 
        np.random.shuffle(indices)
        #if we want to set the batch size of the model to the given batch_size 
        if save_batch_size : 
            self.batch_size = batch_size
        # get the number of batches and the size of the last one. 
        nb_full_batch, last_batch_size  = self.n//batch_size, self.n % batch_size  
        for i in range(nb_full_batch): 
            yield   (self.Y[indices[i*batch_size: (i+1)*batch_size]], 
                    self.covariates[indices[i*batch_size: (i+1)*batch_size]],
                    self.O[indices[i*batch_size: (i+1)*batch_size]], 
                    indices[i*batch_size: (i+1)*batch_size]
                    ) 
        # last batch
        if last_batch_size != 0 : 
            if save_batch_size : 
                self.batch_size = last_batch_size
            yield   (self.Y[indices[-last_batch_size:]], 
                    self.covariates[indices[-last_batch_size:]],
                    self.O[indices[-last_batch_size:]],
                    indices[-last_batch_size:]
                    )
        
    def average_likelihood(self): 
        '''Average the likelihood to smooth it. We do so since we can only estimate
        the likelihood, thus it is random. However, we need it to be accurate since we use this 
        as a stopping criterion. That is why we average it so that it is smoother. 
        '''        
        # we append the last log likelihood computed 
        self.last_likelihoods.append(self.log_like)
        # if we have enough likelihoods to build the mean, we remove the oldest one. 
        if len(self.last_likelihoods) > self.nb_average_likelihood : 
            del(self.last_likelihoods[0])
            
        # we take the mean of the last likelihoods
        self.average_log_like = np.mean(np.array(self.last_likelihoods))        
        #store the average likelihood
        self.log_likelihood_list.append(self.average_log_like)
            
    
    def average_params(self): 
        '''Averages the parameters in order to smooth the variance. 
        We will take, for example, the last self.nb_average_param betas computed to make 
        a better approximation of beta. We will do the same for C. This method adds the 
        last parameters computed to update the mean of the parameter. If we have not 
        enough betas or Sigmas (less than self.nb_average_param), the mean will be 
        on those parameters. 
        Args : 
            None
        Returns : 
            None but update the mean of the last self.nb_average_param parameters.  
        '''
        
        # remove the oldest parameters and add the more recent one.
        self.last_betas[1:self.nb_average_param] = torch.clone(self.last_betas[0: self.nb_average_param-1].detach())
        self.last_betas[0] = torch.clone(self.beta.detach())
        self.last_Cs[1:self.nb_average_param] = torch.clone(self.last_Cs[0: self.nb_average_param-1].detach())
        self.last_Cs[0] = torch.clone(self.C.detach())
        # update the mean of the parameter
        
        # if we have enough parameters 
        if self.cmpt > self.nb_average_param : 
            self.C_mean = torch.mean(self.last_Cs, axis = 0)
            self.beta_mean = torch.mean(self.last_betas, axis = 0)
        
        # if we don't have enough parameters. 
        else : 
            self.cmpt +=1 # to keep track of the number of parameters we have for the mean
            self.C_mean = torch.sum(self.last_Cs, axis = 0)/self.cmpt
            self.beta_mean = torch.sum(self.last_betas, axis = 0)/self.cmpt
        
        


    def fit(self, Y, O, covariates, acc,q,N_epoch_max = 200, lr = 0.015,
            N_iter_max_mode = 100, lr_mode=0.3, VR = 'SAGA', batch_size = 15, 
            class_optimizer = torch.optim.RMSprop, nb_plateau = 15, verbose = False,
           good_init = True):
        '''Batch gradient ascent on the log likelihood given the data. We infer 
        p_theta with importance sampling and then computes the gradients by hand. 
        At each iteration, we look for the right importance sampling law running 
        find_batch_mode. Given this mode, we estimate the variance and we 
        compute the weights required to estimate p_theta. Then, we derive the gradients.
        Note that we only need to know the weights to get gradients. The mean of the 
        weights gives the estimated likelihood that is used as stopping criterion for 
        the algorithm. 
        
        Args : 
               Y: torch.tensor of size (n, p). The counts
               O: torch.tensor of size (n,p). the offset
               covariates: torch.tensor of size (n,p)
               acc: float strictly between. The accuracy you want when computing 
                   the estimation of p_theta. The lower the more accurate but 
                   the slower the algorithm. We will sample 1/acc gaussians 
                   to estimate the likelihood. 
               q: int. The dimension of the latent space. 
               N_epoch_max: int. The maximum number of times we will loop over 
                   the data. We will see N_epoch times each sample. Default is 200. 
               lr: float greater than 0, optional. The learning rate of the batch gradient ascent.
                   Default is 0.015.
               N_iter_mode: int, optional. The maximum iteration you are ready to do to find the 
                   mode of the posterior (i.e. the right importance law). Default is 
                   100. 
               lr_mode: float greater than 0, optional. The learning rate of the gradient ascent
                   we do to find the mode of the posterior. Default is  
               VR : string, optional. the Variance Reductor we want to use. Should be one of those :
                   - 'SAGA'
                   - 'SAG'
                   - 'SVRG' 
                   - None 
                   if None, we are not doing any variance reduction. 
               batch_size : int between 2 and n (included), optional. The batch size of the batch
                   gradient ascent. Default is 15. This is very task-dependant.
               class_optimizer : torch.optim.optimizer object, optional. The optimizer you want to use.
                   Default is torch.optim.RMSprop
               nb_plateau : int, optional. The criterion you want to use. The algorithm
                   will stop if the llikelihood has not increase for nb_plateau iteration 
                   or if we have seen N_epoch_max times each sample. Default is 15. Should not
                   changed (or if changed, should be lower).  
               good_init: Bool. If True, we will do an initialization that is not random. 
                   Takes some time. Default is True. 
       Returns: 
           None, but updates the parameter beta and C. Note that beta_mean 
           and C_mean are more accurate and achieve a better likelihood in general. 
        '''
        self.t0 = time.time() # to keep track of the time
        self.nb_plateau = nb_plateau
        self.batch_size = batch_size
        self.acc = acc
        self.N_samples = int(1/acc) # We will sample 1/acc gaussians to estimate the likelihood.
        t = time.time()
        self.init_data(Y,O, covariates, q, good_init)# initialize the data. 
        #print('time init', time.time()-t)
        self.optim = class_optimizer([self.beta,self.C], lr = lr) # optimizer on C and beta
        self.optim.zero_grad() # We do this since it is possible that beta and C have gradients.
        self.beta = self.beta.to(device)
        self.C = self.C.to(device)
        
        # initialize the Variance Reductors. 
        if VR == 'SAGA' : 
            vr = SAGARAD([self.beta, self.C], self.n)
        elif VR == 'SAG': 
            vr = SAGRAD([self.beta, self.C], self.n)
        elif VR == 'SVRG': 
            vr = SVRGRAD([self.beta, self.C], self.n)
        else : 
            vr = None 
        
        self.max_log_like = -100000000 #to keep track of the best log_likelihood. 
        for j in tqdm(range(N_epoch_max)): 
            # init the log likelihood. We will add each log likelihood of
            # each batch to get the log likelihood of the whole dataset.
            log_like = 0  
            for Y_b, covariates_b, O_b, selected_indices in self.get_batch(batch_size):
                # Store the batches for a nicer implementation.
                self.Y_b, self.covariates_b, self.O_b = Y_b.to(device), covariates_b.to(device), O_b.to(device)
                self.selected_indices = selected_indices
                # compute the log likelihood of the batch and add it to log_likelihood 
                # of the whole dataset.
                # Note that we need to call this function in order to be 
                # able to call self.get_batch_grad_(C/beta)()
                log_like += self.infer_batch_p_theta(N_iter_max_mode, lr_mode).item()
                batch_grad_C = -self.get_batch_grad_C() # add a minus since pytorch minimizes a function. 
                batch_grad_beta = -self.get_batch_grad_beta()
                
                # Given the gradients of the batch, we update the variance 
                # reducted gradients if needed. 
                # Note that we need to give the gradient of each sample in the 
                # batch, not the average gradient of the batch. 
                if vr is not None : 
                    vr.update_new_grad([batch_grad_beta,batch_grad_C], selected_indices)
                else: 
                    self.beta.grad = torch.mean(batch_grad_beta, axis = 0)
                    self.C.grad = torch.mean(batch_grad_C, axis = 0)
                self.optim.step() # optimize beta and C given the gradients.
                #print('time optim.step', time.time()-t)
                self.optim.zero_grad()
                self.average_params() # keep track of some stat
            self.running_times.append(time.time()-self.t0)
            self.log_like = log_like/self.n*batch_size # the log likelihood of the whole dataset.
            self.average_likelihood() # average the  log likelihood for a criterion less random.
            crit = self.compute_criterion(verbose) # compute the criterion 
                                                   # (i.e., check if the likelihood has increased) 
            # If the likelihood has not improved for self.nb_plateau iteration, we stop the algorithm.
            if crit > self.nb_plateau-1:
                print('Algorithm stopped after ', j, ' iterations')
                self.fitted = True
                break 
            # if we use SVRG VR, we need to compute the gradient of the 
            # whole dataset to update the average gradient. 
            if VR == 'SVRG':
                self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
                self.selected_indices = np.arange(0,self.n)
                self.get_gradient_requirement(N_iter_max_mode, lr_mode)
                full_grad_C = -self.get_batch_grad_C()
                full_grad_beta = -self.get_batch_grad_beta()
                vr.update_table([full_grad_beta,full_grad_C])
            # Next, we lower the learning rate (that finds the mode) since after one epoch,
            # we are not very far from the new mode. Indeed, we found the mode for previous 
            # beta and C, and we now need to find the new mode for the new beta and C. We
            # will begin our ascent from the previous mode found. However, since beta and 
            # C won't move very much, the mode for the new beta and C won't move very much
            # either. We can lower the learning rate so that the optimizer will only adjust 
            # a little bit its starting position. 
            
            if j == 0 : 
                lr_mode/=10
            if j == 1 or j == 2: 
                lr_mode/=2
        # The model has been fitted.
        self.fitted = True
        
    def compute_best_loglike(self, acc = 0.001, N_iter_max_mode = 300, lr_mode = 0.001):
        '''Estimate the best likelihood of the model, i.e. the likelihood 
        estimated with beta_mean and C_mean. 
        
        Args: 
            acc: float greater than 0, optional. The accuracy you want for
                the estimation of the likelihood. Default is 0.001.
            N_iter_max_mode : int, optional. The maximum number of iteration you are 
            ready to do to find the mode. Default is 300.  
            lr_mode : float greater than 0. The learning of the gradient ascent 
            you do to find the mode. Default is 0.001. It is small since 
            we are supposed not to bet far from the true mode. 
        Returns : 
            float. The estimated likelihood of beta_mean and C_mean
        '''
        self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
        self.selected_indices = np.arange(0,self.n)
        self.N_samples = int(8/acc)
        # Set beta and C as beta_mean and C_mean to compute the likelihood.
        self.beta = torch.clone(self.beta_mean)
        self.C = torch.clone(self.C_mean)
        # infer p_theta
        self.best_log_like = self.infer_batch_p_theta(N_iter_max_mode, lr_mode)
        return self.best_log_like
        
    def compute_criterion(self, verbose = False):
        '''Updates the criterion of the model. The criterion counts the 
        number of times the likelihood has not improved. We also append 
        the criterion in a list in order to plot it after. 
        Args :
            verbose: bool. If True, will print the criterion whenever it increases.
                Default is True. 
        Returns : int. The criterion. 
        '''
        if self.average_log_like > self.max_log_like : 
            self.max_log_like = self.average_log_like
        else :
            self.crit_cmpt +=1
            if verbose: 
                print(' Criterion : ', self.crit_cmpt , '/', self.nb_plateau)
        self.crit_cmpt_list.append(self.crit_cmpt)
        return self.crit_cmpt 

    def infer_batch_p_theta(self, N_iter_max_mode, lr_mode,take_mean = True): 
        '''Infer p_theta that is computed for a batch of the dataset. The 
        parameter Y,O,cov are in the object itself, so that we don't need 
        to pass them in argument. 
        Args : 
            N_iter_max_mode : int. The number of iteration you are ready 
                to do to find the mode of the posterior.
            lr_mode: postive float. The learning rate of the gradient 
                ascent that finds the mode.
            take_mean: bool. If we want the alogorithm to return 
            the mean of the log likelihood or the log likelihood of each sample.
            Defautl is True. 
        '''
        # Get the gradient requirement. It also computes the weights of the IMPS
        # that are stored in the object. 
        self.get_gradient_requirement(N_iter_max_mode, lr_mode)
        # Take the log of the weights and adjust with the missing constant 
        # self.const that has been removed before to avoid numerical 0. 
        log = torch.log(torch.mean(self.weights,axis = 0))+self.const#*self.mask
        # Return the mean of the log likelihood of the batch if we want to, 
        # or the log of each sample in the batch.  
        if take_mean :
            return torch.mean(log)
        else: 
            return log 
    
    def get_gradient_requirement(self, N_iter_max_mode, lr_mode):
        '''Does all the operation that we need to compute the gradients. 
        We need the gaussian samples and the weights, which we compute here. 
        The gaussians samples needs to be sampled from the right mean and 
        variance, and we find this by calling find_batch_mode and 
        get_batch_best_var methods.
        
        Args: 
            N_iter_mode : int. The maximum number of iterations you are 
                ready to do to find the mode. 
            lr_mode : float greater than 0. The learning rate of the 
                gradient ascent that finds the mode.
        Returns:
            None but computes the weights stored in the object. 
        '''
        # get the mode
        self.find_batch_mode(N_iter_max_mode, lr_mode)
        # thanks to the mode, we can now get the best variance. 
        self.t_other = time.time()
        self.get_batch_best_var()
        # get the samples generated with the mean (mode) and variance found.  
        self.samples = sample_gaussians(self.N_samples, self.batch_mode, self.sqrt_Sigma_b)
        # get the weights
        self.weights = self.get_batch_weights()
        
    def get_batch_weights(self): 
        '''Compute the weights of the IMPS formula. Given the gaussian samples 
        stored in the object,the weights are computed as the ratio of the 
        likelihood of the posterio rand the likelihood of the gaussian samples. 
        Note that we first compute the logarithm of the likelihood of the posterior 
        and the logarithm of the gaussian samples, then remove the maximum of 
        the difference to avoid numerical zero, and takes the exponential. 
        We keep in memory the constant removed to get it back later. 
        
        Args: None 
            
        Returns: torch.tensor of size (N_samples,N_batch). The computed weights. 
        '''
        # Log likelihood of the posterior
        self.log_f = self.batch_un_log_posterior(self.samples)
        # Log likelihood of the gaussian density
        self.log_g = log_gaussian_density(self.samples, self.batch_mode, self.Sigma_b)
        # Difference between the two logarithm
        diff_log = self.log_f-self.log_g 
        self.const = torch.max(diff_log, axis = 0)[0]
        # remove the maximum to avoid numerical zero. 
        diff_log -= torch.max(diff_log, axis = 0)[0]
        weights = torch.exp(diff_log)
        return weights
    def get_batch_best_var(self):
        '''Compute the best variance for the importance law. Given the mode, 
        we can derive the best variance that fits the posterior. Why we do 
        this is a little bit tricky, please see the doc to find out why we do so.
        
        Args: None 
        
        Returns: None but compute the best covariance matrix and 
            its square root, stored in the IMPS_PLN object. 
        '''
        batch_matrix = torch.matmul(self.C.unsqueeze(2), self.C.unsqueeze(1)).unsqueeze(0)
        CW = torch.matmul(self.C.unsqueeze(0),self.batch_mode.unsqueeze(2)).squeeze()
        common = torch.exp(
                           self.O_b  
                           + self.covariates_b@self.beta 
                           + CW
                          ).unsqueeze(2).unsqueeze(3)
        prod = batch_matrix*common
        # The hessian of the posterior
        Hess_post = torch.sum(prod, axis = 1)+torch.eye(self.q).to(device) 
        self.Sigma_b = torch.inverse(Hess_post.detach())
        # Add a term to avoid non-invertible matrix. 
        eps = torch.diag(torch.full((self.q,1),1e-8).squeeze()).to(device)
        self.sqrt_Sigma_b = TLA.cholesky(self.Sigma_b+ eps)
        
    def get_batch_grad_beta(self): 
        ''' Computes the gradient with respect to beta of the log likelihood 
        for the batch. To see why we do so, please see the doc. 
        
        Args: None
        
        Returns: torch.tensor of size (batch_size,d,p). The gradient wrt beta. 
        '''
        first = torch.matmul(self.covariates_b.unsqueeze(2), self.Y_b.unsqueeze(1).double())
        XB = torch.matmul(self.covariates_b.unsqueeze(1),
                          self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(self.C.reshape(1,1,self.p,1,self.q),
                          self.samples.unsqueeze(2).unsqueeze(4)).squeeze()
        Xexp = torch.matmul(self.covariates_b.unsqueeze(0).unsqueeze(3), 
                            torch.exp(self.O_b + XB + CV).unsqueeze(2))
        WXexp =  torch.sum(
                torch.multiply(
                    self.weights.unsqueeze(2).unsqueeze(3), 
                    Xexp), axis = 0)
        sec = WXexp/(torch.sum(self.weights, axis = 0).unsqueeze(1).unsqueeze(2))
        return first-sec
    
    def get_batch_grad_C(self): 
        '''Computes the gradient with respect to C of the log likelihood for 
        the batch. To see why we do so, please see the doc. 
        
        Args: None
        
        Returns: torch.tensor of size (batch_size,d,p). The gradient wrt C.
        ''' 
        XB = torch.matmul(self.covariates_b.unsqueeze(1), self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(
                self.C.reshape(1,1,self.p,1,self.q), 
                self.samples.unsqueeze(2).unsqueeze(4)
                        ).squeeze()
        Ymoinsexp = self.Y_b - torch.exp(self.O_b + XB + CV)
        outer = torch.matmul(Ymoinsexp.unsqueeze(3), self.samples.unsqueeze(2))
        denum = torch.sum(self.weights, axis = 0)
        num = torch.multiply(self.weights.unsqueeze(2).unsqueeze(3), outer)
        batch_grad = torch.sum(num/(denum.unsqueeze(0).unsqueeze(2).unsqueeze(3)), axis = 0)
        return batch_grad
    
    
    def find_batch_mode(self, N_iter_max, lr, eps = 9e-3):
        '''Find the mode of the posterior with a gradient ascent. 
        As starting point, we will use the last mode computed. However, 
        each mode depends on the batch (Y_b,O_b, covariates_b), so that 
        we need to know which batch we are taking. That is why we store
        (in the method .fit()) the current batch took by storing 
        self.selected_indices to know which previous mode to take. 
        
        Args: 
            N_iter_max: int. The maximum number of iteration you are 
                ready to do to find the mode. 
            lr: positive float. The learning of the optimizer for the 
                gradient ascent. 
            eps: positive float. The tolerance. The algorithm will 
                stop if the maximum of |W_t-W_{t-1}| is lower than eps, where W_t 
                is the t-th iteration of the algorithm. Default is 9e-3. 
                This parameter changes a lot the resulting time of the algorithm. 
        Returns : 
            None, but compute and stock the mode in self.batch_mode and the starting point. 
        '''
        # The loss we will use for the gradient ascent. 
        def batch_un_log_posterior(W): 
            return batch_log_P_WgivenY(self.Y_b, self.O_b, self.covariates_b, W, self.C, self.beta) 
        self.batch_un_log_posterior = batch_un_log_posterior
        # get the corresponding starting point. 
        W = self.starting_point[self.selected_indices]
        W.requires_grad = True
        optim = torch.optim.Rprop([W], lr = lr)
        # TODO: need to set an optimizer for the hole dataset, and then optimize only for a batch of them
        # in order to remember the last lr and not do a full gradient ascent 
        criterion = 2*eps
        old_W = torch.clone(W)
        i = 0
        keep_condition = True
        while  i < N_iter_max and keep_condition: 
            loss = -torch.mean(self.batch_un_log_posterior(W))
            # Propagate the gradients
            loss.backward()
            # Update the parameter
            optim.step()
            crit = torch.max(torch.abs(W-old_W))
            optim.zero_grad()
            if crit<eps and i > 2 : 
                keep_condition = False 
            old_W = torch.clone(W)
            i+= 1
        # keep the number of iteration as information 
        self.nb_iteration_list.append(i)
        # stock the mode
        self.batch_mode = torch.clone(W.detach())
        # stock the starting point for the next epoch. 
        W_requires_grad = False
        self.starting_point[self.selected_indices] = torch.clone(W.detach())
        
    def get_Sigma(self): 
        '''Get Sigma by computing C_mean@C_mean^T. '''
        return (self.C_mean.detach())@(self.C_mean.detach().T)
    
    def show_Sigma(self): 
        '''Displays Sigma'''
        sns.heatmap(self.get_Sigma())
        plt.show()
        
    def __str__(self):
        '''Show the model, Sigma and the likelihood.'''
        try : 
            self.best_log_like = torch.max(self.log_likelihood_list) 
            print('Log likelihood of the model: ', self.best_log_like) 
        except :  
            print('Please fit the model')
            raise AttributeError
        show(self)
        self.show_Sigma()
        return ''
    
    def show(self, save = False, name_graphic = '', display_best_log_like = False): 
        """Show some useful stats of the model. It will plot the estimated log_likelihood
        and the criterion in the y axis with the runtime in the x-axis. The model 
        should have been fitted. 

        Args : 
            save: bool, optional. If True, the graphic will be saved. If false, won't be saved.
                default is False. 
            name_doss: str, optional. The name of the file you want to save the graphic. Default is 
                an empty string.
            display_best_log_like: bool, optional. If True, we will compute and display the likelihood of 
                best parameters. We do so  since computing the likelihood can be very demanding.
                Default is False. 
        Returns : 
                None but displays the figure. It can also save the figure if save = True. 
        Raises: 
            AttributeError when the model has not been fitted. 
        """
        # Make sure the model has been fitted
        if not self.fitted: 
            print('Please fit the model before by calling model.fit(Y,O,covariates,N_epoch,acc)')
            raise AttributeError 
        else : 
            fig,ax = plt.subplots(2,1,figsize = (10,8))
            abscisse = self.running_times
            print('max likelihood :', np.max(np.array(self.log_likelihood_list)))
            # plot the negative likelihood of the model
            ax[0].plot(np.arange(0, len(self.log_likelihood_list)), 
                       -np.array(self.log_likelihood_list))
            ax[0].set_title('Negative log likelihood')
            ax[0].set_ylabel('Negative loglikelihood')
            ax[0].set_xlabel('Seconds')
            if display_best_log_like :
                try : 
                    best_log_like = self.best_log_like
                except : 
                    best_log_like = self.compute_best_loglike(acc = 0.001).item()            
                ax[0].axhline(-best_log_like, c = 'red', 
                              label = 'Negative likelihood of the best parameters')
            ax[0].set_yscale('log')
            ax[0].legend()
            
            # plot the criteria of the model
            ax[1].plot(np.arange(0, len(self.crit_cmpt_list)),self.crit_cmpt_list)
            ax[1].set_title('Number of epoch the likelihood has not improved')
            ax[1].set_xlabel('Seconds')
            ax[1].legend()
            #save the graphic if needed
            if save : 
                plt.savefig(name_graphic)
            plt.show()
    
#################################### FastPLN object #######################################################
    
    
def ELBO(Y, O,covariates ,M ,S ,Sigma ,beta):
    '''Compute the ELBO (Evidence LOwer Bound. See the doc for more details 
    on the computation
    
    Args: 
        Y: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        Sigma: torch.tensor. Model parameter with size (p,p)
        beta: torch.tensor. Model parameter with size (d,p)
    Returns: 
        torch.tensor of size 1, with a gradient. The ELBO. 
    '''
    n,p = Y.shape
    SrondS = torch.multiply(S,S)
    OplusM = O+M
    MmoinsXB = M-torch.mm(covariates, beta) 
    tmp = torch.sum( torch.multiply(Y, OplusM)  
                     -torch.exp(OplusM+SrondS/2) 
                     +1/2*torch.log(SrondS)
                   )
    DplusMmoinsXB2 = torch.diag(torch.sum(SrondS, dim = 0))+ torch.mm(MmoinsXB.T, MmoinsXB)
    tmp -= 1/2*torch.trace(  
                            torch.mm(  
                                        torch.inverse(Sigma), 
                                        DplusMmoinsXB2
                                    )
                          )
    tmp-= n/2*torch.log(torch.det(Sigma))
    return tmp


class fastPLN():
    '''Implement the variational algorithm infering the parameters of the PLN model,
    with a closed form for the M step and a gradient step for the VE step. Any value of n 
    and p can be taken.
    '''
    def __init__(self): 
        '''Defines some usefuls lists and variables for the object. A deeper initalization is done 
        in the init_data() func, once the dataset is available.
        '''
        self.old_loss = 1
        # some lists to store some stats
        self.running_times = list()
        self.deltas = list()
        self.norm_grad_M = list()
        self.norm_grad_S = list()
   
    def init_data(self, Y, O, covariates, good_init): 
        '''Initialize the parameters with the right shape given the data. 
        
        Args: 
              data: list with 3 elements(torch.tensor): Y, O and covariates in this 
              order. Y and O should be of size (n,p), covariates of size (n,d). 
        Returns:
            None but initialize some useful data. 
        '''
        #known variables
        try : 
            self.Y = torch.from_numpy(Y).to(device)
            self.O = torch.from_numpy(O).to(device)
            self.covariates = torch.from_numpy(covariates).to(device)
        except : 
            self.Y = Y.to(device)
            self.O = O.to(device)
            self.covariates = covariates.to(device)
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        if good_init : 
            print('Initialization ...') 
            #model parameter 
            poiss_reg = Poisson_reg()
            poiss_reg.fit(self.Y, self.O, self.covariates)
            self.beta = torch.clone(poiss_reg.beta.detach()).to(device)
            self.Sigma =  init_Sigma(self.Y, self.O, self.covariates, self.beta).to(device) 
            # Initialize C in order to initialize M. 
            self.C = torch.cholesky(self.Sigma).to(device)
            self.M = init_M(self.Y, self.O, self.covariates, self.beta,self.C, 300, 0.1) 
            self.M.requires_grad_(True)
            #self.M+= covariates@self.beta
            print('Initialization finished')
        else:
            self.beta = torch.randn((self.d,self.p))
            self.Sigma = torch.diag(torch.ones(self.p))
            self.M = torch.ones(self.n, self.p)

        # No better initialization possible for S
        self.S = 1/2*torch.ones((self.n,self.p)).to(device)
        
    
    def compute_ELBO(self): 
        '''Compute the ELBO with the parameter of the model.'''
        return ELBO(self.Y,self.O , self.covariates,self.M ,self.S ,self.Sigma ,self.beta)
    
    
    def fit(self,Y,O,covariates, N_iter_max= 200, tol_delta = None,tol_RMSE_M_grad =  0.1  ,
            optimizer = torch.optim.Rprop, lr = 0.1,good_init = True, verbose = False): 
        '''Main function of the class. Infer the best parameter Sigma and beta given the data.
        
        Args:
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            N_iter_max: int. The maximum number of iteration you are ready to do.
                Default is 200. 
            tol_delta: non negative float. Criterion for the model (Default is None).
                If tol_delta is None, then we will set it as 1/(2*p). If the RMSE 
                between the old parameters and the new ones are lower than tol_delta, 
                the algorithm will stop. 
            tol_RMSE_M_grad: non negative float. Criterion for the model (Default is 0.1). 
                If the RMSE of the gradient of M is lower than tol_RMSE_M_grad, the 
                algorithm will stop. 
            optimizer: objects that inherits from torch.optim. The optimize you want. 
                Default is torch.optim.Rprop.
            lr: positive float. The learning rate of the optimizer. Default is 0.1.
            good_init: bool. If True, we will do a smart initialization instead of 
                a random initialization. Default is True. 
            verbose: bool. If True, will print some stats during the fitting. Default is 
                False. 
            
        Returns: 
            None but update the parameter C and beta of the object.
        '''
    
        
        self.t0 = time.time()
        #initialize the data
        self.init_data(Y,O,covariates, good_init)
        if tol_delta is None : 
            tol_delta = 1/(2*self.p)
        
        self.optimizer = optimizer([self.S,self.M], lr = lr)
        stop_condition = False 
        i = 0
        self.old_beta = torch.clone(self.beta.detach())
        self.old_Sigma = torch.clone(self.Sigma.detach())
        while i < N_iter_max and stop_condition == False: 
            # VE step
            self.optimizer.zero_grad()
            self.M.grad = -self.grad_M()
            self.S.grad = -self.grad_S()
            self.optimizer.step()
            # M step
            self.beta = self.closed_beta()
            self.Sigma = self.closed_Sigma()
            # Criterions
            delta_beta = RMSE(self.old_beta-self.beta).item()
            delta_Sigma = RMSE(self.old_Sigma-self.Sigma).item()
            delta = delta_beta + delta_Sigma
            RMSE_M_grad = RMSE(self.M.grad)
            
            # condition to see if we have reached the tolerance threshold
            if  delta < tol_delta or RMSE_M_grad<tol_RMSE_M_grad:
                stop_condition = True 
            self.old_Sigma = torch.clone(self.Sigma.detach())
            self.old_beta = torch.clone(self.beta.detach())
            # print some stats if we want to
            if i%10 == 0 and verbose : 
                print('Iteration number: ', i)
                print('-------UPDATE-------')
                print('Delta : ', delta)
            i += 1
            # keep records
            self.keep_records(delta, RMSE_M_grad)
            
            
        if stop_condition : 
            print('Last delta: {}, last_RMSE_M : {} reached in {} iterations'.format(delta,RMSE_M_grad, i))
        else : 
            print('Maximum number of iterations reached : ', N_iter_max, 'last delta = ', delta)

    def keep_records(self, delta, RMSE_M_grad):
        '''Keep some records to plot after.'''
        self.norm_grad_M.append(RMSE_M_grad)
        self.norm_grad_S.append(RMSE(self.S.grad))
        self.deltas.append(delta)
        
        self.running_times.append(time.time()-self.t0)
    def grad_M(self):
        '''Compute the gradient of the ELBO with respect to M'''
        grad = self.Y - torch.exp(self.O+self.M+torch.multiply(self.S,self.S)/2)
        grad -= torch.mm(self.M-torch.mm(self.covariates,self.beta), torch.inverse(self.Sigma))
        return grad 

    def grad_S(self): 
        '''Compute the gradient of the ELBO with respect to S'''
        grad = torch.div(1,self.S)
        grad -=torch.multiply(self.S, torch.exp(self.O+self.M+torch.multiply(self.S,self.S)/2))
        grad -=torch.mm(self.S, torch.diag(torch.diag(torch.inverse(self.Sigma))))
        return grad

    def closed_Sigma(self):
        '''Closed form for Sigma for the M step.'''
        n,p = self.M.shape
        MmoinsXB = self.M-torch.mm(self.covariates,self.beta)
        
        closed = torch.mm(MmoinsXB.T,MmoinsXB)
        closed +=torch.diag(torch.sum(torch.multiply(self.S,self.S), dim = 0))
        return 1/(n)*closed
    def closed_beta(self): 
        '''Closed form for beta for the M step.'''
        ## a améliorer l'inverse !
        
        return torch.mm(
                        torch.mm(
                                torch.inverse(torch.mm(
                                                    self.covariates.T,
                                                    self.covariates)),
                                self.covariates.T),
                        self.M)

    def show_Sigma(self):
        '''Simple method that displays Sigma to see the global structure.'''
        sns.heatmap(self.Sigma.detach().numpy())
        plt.show()
    def init_M(self, N_iter, lr, eps = 7e-3):
        '''Initialization for the variational parameter M. Basically, we take 
        the mode of the log_posterior as initialization.
        
        Args: 
            N_iter: int. The maximum number of iteration you are ready to 
                do to find the mode. 
            lr: positive float. The learning rater of the optimizer. A good 
        '''
        def batch_un_log_posterior(W): 
            return batch_log_P_WgivenY(self.Y, self.O, self.covariates,  W, self.C, self.beta) 
        self.batch_un_log_posterior = batch_un_log_posterior
        W = torch.randn(self.n,self.p)
        W.requires_grad_(True)
        optimizer = torch.optim.Rprop([W], lr = lr)
        criterion = 2*eps
        old_W = torch.clone(W)
        keep_condition = True
        while  i < N_iter_max and keep_condition: 
            loss = -torch.mean(self.batch_un_log_posterior(W))
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
    
    
    def show(self,name_doss = '' ,  save = False):
        '''displays some useful stats of the model. 

        args : 
            'model' fastPLN object where you have called model.fit_IMPS()
            'name_doss' : str. the name of the file you want to save the graphic. 
            'save' : bool. If True, the graphic will be saved. If false, won't be saved. 

        returns : 
                None but displays the figure. It can also save the figure if save = True. 
        '''
        fig,ax = plt.subplots(3,1,figsize = (15,15))
        abscisse = model.running_times
        plt.subplots_adjust(hspace = 0.4)

        # plot the L1 norm of the gradients. 
        ax[0].plot(abscisse, model.deltas, label = 'Deltas')
        ax[0].set_title('Delta Criteria')
        ax[0].set_yscale('log')
        ax[0].legend()


        ax[1].plot(abscisse, model.norm_grad_M, label = ' norm grad M ')
        ax[1].set_yscale('log')
        ax[1].legend()

        ax[2].plot(abscisse, model.norm_grad_S, label = ' norm grad S')
        ax[2].set_yscale('log')
        ax[2].legend()
        
        if save : 
            plt.savefig(name_doss)
        plt.show()
        
        
######################################### fastPLNPCA object ##############################        
        
def ELBO_PCA(Y, O,covariates ,M ,S ,C ,beta):
    '''compute the ELBO with a PCA parametrization'''
    # Store some variables that will need to be computed twice 
    A = O + torch.mm(covariates,beta)+torch.mm(M,C.T)
    SrondS = torch.multiply(S,S)
    # Next, we add the four terms of the ELBO_PCA 
    first = torch.sum(torch.multiply(Y,A))
    second = torch.sum(-1/2*torch.exp(A + 1/2*torch.mm(SrondS,torch.multiply(C,C).T)))
    third = 1/2*torch.sum(torch.log(SrondS))
    fourth = torch.sum(-1/2*(torch.multiply(M,M)+torch.multiply(S,S)))
    return first +second + third + fourth 

class fastPLNPCA():
    def __init__(self): 
        '''Defines some usefuls lists and variables for the object. A deeper initalization is done 
        in the init_data() func, once the dataset is available.
        '''
        self.old_loss = 1
        # some lists to store some stats
        self.ELBO_list = list()
        self.running_times = list()
        self.deltas = list()
        
    def init_data(self,Y, O, covariates, q, good_init): 
        '''Initialize the parameters with the right shape given the data. 
        
        Args: 
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            q: int. The dimension of the latent space. 
            good_init: bool. Tells if we want to do a good initialization (not random).
                  Takes some time. 
        Returns:
            None but initialize some useful data.'''
        self.Y = Y.to(device)
        self.O = O.to(device)
        self.covariates = covariates.to(device)
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        self.q = q 
        
        # If we want to do a good initialization
        if good_init : 
            poiss_reg = Poisson_reg()
            poiss_reg.fit(self.Y,self.O,self.covariates)
            # Model parameter 
            self.beta = torch.clone(poiss_reg.beta.detach()).to(device) 
            self.C = init_C(self.Y, self.O, self.covariates, self.beta, self.q).to(device)
            # Variational parameter
            self.M = init_M(self.Y, self.O, self.covariates, self.beta,self.C, 300, 0.1)
        else: 
            self.C = torch.randn((self.p,self.q)).to(device)
            self.beta = torch.randn((self.d,self.p)).to(device)
            self.M = torch.randn((self.n,self.q)).to(device) 
        # Can't do any further initialization
        self.S = torch.randn((self.n,self.q)).to(device)
        # Set some gradients for optimization
        self.beta.requires_grad_(True)
        self.M.requires_grad_(True)
        self.S.requires_grad_(True)
        self.C.requires_grad_(True)
        
    
    def compute_ELBO_PCA(self):
        '''Compute the ELBO of the PCA parametrisation with the parameter of the model.'''
        return ELBO_PCA(self.Y,self.O , self.covariates,self.M ,self.S ,self.C ,self.beta)
    
    
    def fit(self, Y, O, covariates, q,N_iter_max = 8000, tolerance = 0.0002, optimizer = torch.optim.Rprop,
            lr = 0.1, good_init = True, verbose = False):
        '''Main function of the class. Infer the best parameter C and beta given the data.
        
        Args:
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            N_iter_max: int. The maximum number of iteration you are ready 
                to do (Default is 8000).  
            tolerance: non negative float. Criterion for the model (Default is 0.0002).
                If the RMSE between the old parameters and the new ones are 
                lower than tol_delta, the algorithm will stop. 
            optimizer: objects that inherits from torch.optim. The optimize you want. 
                Default is torch.optim.Rprop.
            lr: positive float. The learning rate of the optimizer. Default is 0.1.
            good_init: Bool. If True, will do a good initialization. Takes some time. 
                Default is True. 
            verbose: bool. If True, will print some stats during the fitting. Default is 
                False. 
        Returns: 
            None but update the parameter C and beta of the object.
        '''
        self.t0 = time.time()
        #initialize the data
        self.init_data(Y, O, covariates, q, good_init)
        self.optimizer = optimizer([self.beta, self.C, self.M, self.S], lr = lr)
        stop_condition = False 
        i = 0
        while i < N_iter_max and stop_condition == False : 
            self.optimizer.zero_grad()
            loss = -self.compute_ELBO_PCA()
            loss.backward()
            self.old_beta = torch.clone(self.beta)
            self.old_Sigma = torch.clone(self.get_Sigma())
            self.optimizer.step()
            # Compute the criterion
            delta_beta = RMSE(self.old_beta - self.beta).item() # precision
            delta_Sigma = RMSE(self.old_Sigma - self.get_Sigma()).item()
            delta = delta_beta + delta_Sigma
            self.deltas.append(delta)
            # Condition to see if we have reach the tolerance threshold
            if  abs(delta) < tolerance :
                stop_condition = True 
            self.ELBO_list.append(-loss.item())# keep track of the ELBO
            # Print some stats if we want to
            if i%100 == 0 and verbose : 
                print('Iteration number: ', i)
                print('-------UPDATE-------')
                print('Delta : ', delta)
            # Keep track of the time 
            self.running_times.append(time.time()-self.t0)
            i+=1

        if stop_condition : 
            print('Tolerance {} reached in {} iterations'.format(tolerance, i))
        else : 
            print('Maximum number of iterations reached : ', N_iter_max, 'last delta = ', delta)

    def get_Sigma(self): 
        '''Return the parameter Sigma of the model, that is CC^T'''
        return self.C@(self.C.T)
    
    def gradPCA_beta(self):
        '''Compute the gradient of the ELBO with respect to beta. Sanity check'''
        matC  = self.C 
        CrondC = torch.multiply(matC,matC)
        SrondS = torch.multiply(self.S,self.S)
        first_term = torch.mm(self.covariates.T,self.Y)
        second_term = -1/2*torch.mm(
                             self.covariates.T,
                             torch.exp(self.O+torch.mm(
                                                       self.covariates,
                                                       self.beta
                                                      )
                                         +torch.mm(self.M,matC.T)+1/2*torch.mm(SrondS,CrondC.T))
                             )
        return first_term+second_term
    
    def gradPCA_M(self):
        '''Compute the gradient of the ELBO with respect to M. Sanity check'''
        CrondC = torch.multiply(self.C,self.C)
        SrondS = torch.multiply(self.S,self.S)
        A = self.O +torch.mm(self.covariates,self.beta) + torch.mm(self.M,self.C.T)
        first = torch.mm(self.Y,self.C)
        second = -1/2*torch.mm(torch.exp(A+1/2*torch.mm(SrondS,CrondC.T)),self.C)
        third = -self.M
        return first + second+third
    def gradPCA_S(self):
        '''Compute the gradient of the ELBO with respect to S. Sanity check'''
        matC = self.C
        CrondC = torch.multiply(matC,matC)
        SrondS = torch.multiply(self.S,self.S)
        A = self.O +torch.mm(self.covariates,self.beta) + torch.mm(self.M,matC.T)
        first = -1/2*torch.multiply(self.S,torch.mm(torch.exp(A+1/2*torch.mm(SrondS,CrondC.T)),CrondC))
        second = torch.div(1,self.S)
        third = -self.S
        return first + second +third
    
    def gradPCA_C(self):
        '''Compute the gradient of the ELBO with respect to C. Sanity check'''
        matC  = self.C 
        CrondC = torch.multiply(matC,matC)
        SrondS = torch.multiply(self.S,self.S)
        first = torch.mm(self.Y.T,self.M) 
        A = self.O +torch.mm(self.covariates,self.beta) + torch.mm(self.M,matC.T)
        exp = torch.exp(A + 1/2*torch.mm(SrondS,CrondC.T))
        second = -1/2*torch.mm(exp.T,self.M)-1/2*torch.multiply(matC,torch.mm(exp.T,SrondS))
        return first + second

    def show(self,name_doss='Stat_model', save = False):
        '''displays some useful stats of the model. 

        args : 
            'model' fastPLNPCA object where you have called model.fit_IMPS()
            'name_doss' : str. the name of the file you want to save the graphic.
                Default is 'Stat_model'. 
            'save' : bool. If True, the graphic will be saved. If false, won't be saved. 

        returns : 
                None but displays the figure. It can also save the figure if save = True. 
        '''

        fig,ax = plt.subplots(2,1,figsize = (15,12))
        max_ = np.max(np.array(self.ELBO_list))
        print('Maximum of the ELBO', max_)

        abscisse = self.running_times
        plt.subplots_adjust(hspace = 0.4)
        # Plot the negative ELBO minus the maximum for a nice plot
        ax[0].plot(abscisse, np.log(-np.array(self.ELBO_list)+max_), label = 'ELBO')
        ax[0].legend()
        ax[0].set_title('ELBO')
        ax[0].set_ylabel('ELBO')
        ax[0].set_xlabel('Seconds')
        
        # Plot the criteria of the algorithm. 
        ax[1].plot(abscisse, self.deltas, label = 'deltas')
        ax[1].set_yscale('log')
        ax[1].legend()

        ax[0].legend()
        if save : 
            plt.savefig(name_doss)
        plt.show()
        sns.heatmap(self.get_Sigma().detach())
        plt.show()




    
   



