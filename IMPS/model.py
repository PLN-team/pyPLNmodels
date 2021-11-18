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

from .utils import init_C, Poisson_reg, log_stirling
from .VRA import SAGARAD, SAGRAD, SVRGRAD 

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')
print('device ', device)


def show(model, save = False, name_graphic = '', display_best_log_like = False): 
        """
        Show some useful stats of the model. It will plot the estimated log_likelihood
        and the criterion in the y axis with the runtime in the x-axis.  

        Args : 
            model: IMPS_PLN object where model.fit(Y,cov, O) has already been called. 
            name_doss: str. The name of the file you want to save the graphic. 
            save: bool. If True, the graphic will be saved. If false, won't be saved. 
            display_best_log_like: bool. If True, we will compute and display the likelihood of 
                best parameters. We do so  since computing the likelihood can be very demanding. 
        Returns : 
                None but displays the figure. It can also save the figure if save = True. 
        Raises: 
            AttributeError when the model has not been fitted. 
        """
        
        # Make sure the model has been fitted
        if not model.fitted: 
            print('Please fit the model before by calling model.fit(Y,O,covariates,N_epoch,acc)')
            raise AttributeError 
        else : 
                
            
            fig,ax = plt.subplots(2,1,figsize = (10,8))
            abscisse = model.running_times
            
            # plot the negative likelihood of the model
            ax[0].plot(np.arange(0, len(model.log_likelihood_list)), 
                       -np.array(model.log_likelihood_list))
            ax[0].set_title('Negative log likelihood')
            ax[0].set_ylabel('Negative loglikelihood')
            ax[0].set_xlabel('Seconds')
            if display_best_log_like :
                try : 
                    best_log_like = model.best_log_like
                except : 
                    best_log_like = model.compute_best_loglike(acc = 0.001).item()            
                ax[0].axhline(-best_log_like, c = 'red', 
                              label = 'Negative likelihood of the best parameters')
            ax[0].set_yscale('log')
            ax[0].legend()
            
            # plot the criteria of the model
            ax[1].plot(np.arange(0, len(model.crit_cmpt_list)),model.crit_cmpt_list)
            ax[1].set_title('Number of epoch the likelihood has not improved')
            ax[1].set_xlabel('Seconds')
            ax[1].legend()
            #save the graphic if needed
            if save : 
                plt.savefig(name_graphic)
            plt.show()

            
def log_likelihood(Y,O,covariates, C,beta, acc = 0.002, N_iter_mode = 300, lr_mode= 0.1): 
    """Estimate the log likelihood of C and beta given Y,O,covariates. 
    The process is a little bit complicated since we need to find 
    the mode of the posterior in order to sample the right Gaussians. 
    """
    
    q = C.shape[1]
    model = IMPS_PLN(q)
    model.init_data(Y,O,covariates)
    model.Y_b, model.O_b, model.covariates_b = model.Y, model.O, model.covariates
    model.C_mean = C
    model.beta_mean = beta
    log_like = model.compute_best_loglike(acc, N_iter_mode, lr_mode)
    return log_like
 
            
def sample_gaussians(N_samples, mean,sqrt_Sigma):
    '''
    samples some gaussians with the right mean and variance. Be careful, we ask for the square root of Sigma, not Sigma. will detect automatically if we are using batches or not.
    
    args : 
         N_samples : int. the number of samples you want to sample. 
         mean : torch.tensor of size (n_batch,q) or (q)
         sqrt_Sigma : torch.tensor or size (n_batch, q, q) or (q,q)
         
    out : 
        a torch.tensor of size (N_samples, mean.shape). It is a vector of N_samples gaussian of dimension mean.shape. For each  1< i< N_sample, 1<k< n_batch , out[i,k] is a gaussian with mean mean[k,:] and variance sqrt_Sigma[k,:,:]@sqrt_Sigma[k,:,:].  
    '''
    # if we must do the matrix multiplication along the right axis, so that we need to check the dimension of the entry mean. Note that we use mainly the first case. 
    if len(mean.shape)> 1 : 
        q = mean.shape[1]
        W_orig = torch.randn(N_samples, 1,q,1).to(device)
    elif len(mean.shape) == 1 : 
        q = mean.shape[0]
        W_orig = torch.randn(N_samples, q, 1).to(device)
    # just add the mean and multiply by the square root matrice to sample from the right distribution. 
    W = torch.matmul(sqrt_Sigma.unsqueeze(0), W_orig).squeeze() + mean.unsqueeze(0)
    return W

def log_gaussian_density(W, mu_p,Sigma_p): 
    '''
    computes the log density of a gaussian W of size (N_samples, n_batch, q) or (N_samples, q) With mean mu_p and Sigma_p.
    
    args : W: torch.tensor of size (N_samples, n_batch, q) or (N_samples, q)
          mu_p : torch.tensor : the mean from which the gaussian has been sampled
          Sigma_p : torch.tensor. The variance from which the gaussian has been sampled. 
    returns : 
            torch.tensor. the log of the density of W, taken along the last axis.  
    '''
    dim = W.shape[-1] # dimension q
    const = torch.sqrt((2*math.pi)**dim*torch.det(Sigma_p)) ## constant of the gaussian density
    Wmoinsmu = W-mu_p.unsqueeze(0)
    inv_Sig = torch.inverse(Sigma_p)
    # we need to do the matrix multiplication along the right axis so that we need to check if the entry has a dimension for the batch. 
    if len(mu_p.shape)>1 : 
        log_d = -1/2*torch.matmul(torch.matmul(inv_Sig.unsqueeze(0), Wmoinsmu.unsqueeze(3)).squeeze().unsqueeze(2), Wmoinsmu.unsqueeze(3))
    elif len(mu_p.shape)== 1 :  
        log_d = -1/2*torch.matmul(Wmoinsmu.unsqueeze(1),torch.matmul(inv_Sig.unsqueeze(0), Wmoinsmu.unsqueeze(2))).squeeze()
    return log_d.squeeze() - torch.log(const)



def batch_log_P_WgivenY(Y_b, covariates_b, O_b, W, C, beta): 
    '''
    computes the log posterior of the PLN model. we compute it either for W of size (N_samples, N_batch,q) or (N_samples, q) 
    
    args : 
        Y_b : torch.tensor of size (N_batch, p) 
        covariates_b : torch.tensor of size (N_batch, d) or (d) 
    
    does the same as log_P_W_givenY but add one more dimension for computing this for a batch instead of only one element. 
    '''
    length = len(W.shape)
    q = W.shape[-1]
    if length == 2 : 
        CW = torch.matmul(C.unsqueeze(0),W.unsqueeze(2)).squeeze()
    elif length == 3 : 
        CW = torch.matmul(C.unsqueeze(0).unsqueeze(1), W.unsqueeze(3)).squeeze()
    A_b = O_b + CW + covariates_b@beta
    return -q/2*math.log(2*math.pi)-1/2*torch.norm(W, dim = -1)**2 + torch.sum(-torch.exp(A_b)   + A_b*Y_b - log_stirling(Y_b) , axis = -1)


class PLN():
    '''
    Class that maximizes the likelihood of the PLN-PCA model. The main function is the method fit() that fits the model. Most of the others functions are here to support the fit method. 
    '''
    
    def __init__(self, q,nb_average_param = 100, nb_average_likelihood = 8, nb_plateau = 10):
        '''
        init method. 
        
        args : 
            q : int. The dimension you want for the PLN-PCA model. 
            nb_average_param : int. We will average the parameter to get parameters with lower variance.
                                nb_average param tells the number of parameter we take to build the mean. Should
                                not be changed since not very important. 
            nb_average_likelihood : int. We will average the log_likelihood of the model. nb_average likelihood
                                    tells the number of likelihood we will take to build the mean likelihood. 
                                    Should not be changed since not very important. Note that this parameter is actually 
                                    changing the algorithm (just a little bit) since the stopping criterion
                                    depends directly on the average likelihood. 
        returns : an IMPS_PLN object. 
        '''
        self.q = q # the dimension of the latent space 
        self.nb_average_likelihood = nb_average_likelihood
        self.nb_average_param = nb_average_param 
        self.running_times = list() # list to store the running times for a nice plot
        self.log_likelihood_list = list() # list to store the likelihood to plot it after
        self.last_likelihoods = list() # list that will store the last likelihoods in order to take the mean of those likelihoods to smooth it. 
        self.criteria_list = list() # list that will store all the criterion. 
        self.nb_iteration_list = list() # list that will store the number of iteration we are doing at for each gradient ascent
                                        # that finds the mode. 
        
    def init_data(self, Y,O,covariates): 
        '''
        Initialise some usefuls variables given the data. 
        We also initialise C and beta. 
        
        args : 
               Y : torch.tensor of size (n, p). The counts
               O : torch.tensor of size (n,p). the offset
               covariates : torch.tensor of size (n,p) 
        returns : 
                None
        '''
        self.fitted = False # bool variable toknow if we have fitted the model. 
        self.cmpt = 0 # variable that counts some iterations. 
        self.crit_cmpt = 0 
        self.crit_cmpt_list = list()
        #data 
        self.Y = Y.float().to(device)
        self.covariates = covariates.to(device)
        self.O = O.to(device)
        
        self.n = Y.shape[0] 
        self.p = Y.shape[1]
        self.d = self.covariates.shape[1]
        
        self.starting_point = torch.zeros(self.n, self.q) # tensor that will store the starting point for the 
                                                          # gradient descent that finds the mode for IMPS
        self.last_betas = torch.zeros(self.nb_average_param,self.d,self.p) # init of the average of the last betas
        self.last_Cs = torch.zeros(self.nb_average_param, self.p, self.q) #init of the average of the last Sigmas
        
        # initialization for beta with a poisson regression 
        poiss_reg = Poisson_reg()
        poiss_reg.fit(Y,O,covariates)
        self.beta = torch.clone(poiss_reg.beta.detach()).to(device)
        
        
        # initialization for C with an array of size (p,q) taking the q vectors associated to 
        # the q largest eigenvectors of the estimated variance of log(Y)
        self.C = init_C(O, covariates, Y, self.beta, self.q).to(device)
                
        #setting some gradients for optimization. 
        self.C.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.C_mean = torch.clone(self.C)
        self.beta_mean = torch.clone(self.beta)
        
    def get_batch(self,batch_size, save_batch_size = True): 
        '''
        get the batches required to do a  minibatch gradient ascent.  
        
        args : 
                'batch_size' int.  the batch size you want. 
                
        returns : a generator. Will generate n/batch_size samples of size batch_size (except the last one 
                    since the rest of the division is not always 0)
        '''
        indices = np.arange(self.n)
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
        if last_batch_size != 0 : 
            if save_batch_size : 
                self.batch_size = last_batch_size
            yield   (self.Y[indices[-last_batch_size:]], 
                    self.covariates[indices[-last_batch_size:]],
                    self.O[indices[-last_batch_size:]],
                    indices[-last_batch_size:]
                    )
            
        

    def keep_records(self): 
        '''
        function that keep some records in order to plot the evolution after.
        args : None
        returns : None
        '''
        # we average the parameters
        self.average_params()
        # keep record of the running time
        self.running_times.append(time.time()-self.t0)

    def average_likelihood(self): 
        '''
        average the likelihood to smooth it. We do so since we can only estimate
        the likelihood, so that it is random. However, we need it to be accurate since we use this 
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
        '''
        method that averages the parameters in order to smooth the variance. 
        We will take, for example, the last self.nb_average_param betas computed to make 
        a better approximation of beta. We will do the same for C.
        This function adds the last parameters computed to update the mean of the parameter. 
        If we have not enough betas or Sigmas (less than self.nb_average_param), the mean will be on those parameters. 
        args : 
            log_like : the likelihood computed with the current parameters.
        
        returns : 
                None but update the mean of the last self.average parameters.  
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
        
        


    def fit(self, Y, O, covariates,  N_epoch, acc,lr = 0.008,N_iter_mode = 100, lr_mode=0.3, VR = 'SAGA', batch_size = 10, class_optimizer = torch.optim.RMSprop, nb_plateau = 15):
        '''
        Does a (batch) gradient ascent on the log likelihood given the data. We infer p_theta with importance sampling and then computes the gradients by hand. 
        at each iteration, we look for the right importance sampling law by running find_batch_mode. Given this law, 
        we compute the weights required to estimate p_theta. Then, we derive the gradients. Note that we only need to 
        know the weights to get gradients. The mean of the weights gives the estimated likelihood that is used as stopping criterion for the algorithm. 
        
        args : 
               Y : torch.tensor of size (n, p). The counts
               O : torch.tensor of size (n,p). the offset
               covariates : torch.tensor of size (n,p)
               N_epoch : int. The number of times we will loop over the data. We will see N_epoch times each sample
               acc : float between 1 and 0 (strictly in this interval). The accuracy you want when computing the estimation of p_theta. The lower the more accurate but the slower the algorithm. We will sample 1/acc gaussians to estimate the likelihood. 
               lr : float greater than 0. The learning rate of the batch gradient ascent.
               N_iter_mode : the maximum iteration you are ready to do to find the mode of the posterior (i.e. the right importance law) 
               lr_mode : float greater than 0. The learning rate of the gradient ascent we do to find the mode of the posterior. 
               VR : string. the Variance Reductor we want to use. Should be one of those :
                   - 'SAGA'
                   - 'SAG'
                   - 'SVRG' 
                   - None 
                   if None, we are not doing any variance reduction. 
               batch_size : int between 2 and n (included) the batch size of the batch gradient ascent. 
               class_optimizer : torch.optim.optimizer object. The optimizer you want to use. 
               nb_plateau : the criterion you want to use. The algorithm will stop if the llikelihood has not increase 
                           for nb_plateau iteration or if we have seen N_epoch times each sample. 
       
       returns : 
           None, but updates the parameter beta and C. Note that beta_mean and C_mean are more accurate and achieve a better likelihood in general. 
        '''
        self.t0 = time.time() # to keep track of the time
        self.nb_plateau = nb_plateau
        self.batch_size = batch_size
        self.acc = acc
        self.N_samples = int(1/acc) # We will sample 1/acc gaussians
        self.init_data(Y,O, covariates)# initialize the data. 
        self.optim = class_optimizer([self.beta,self.C], lr = lr) # optimizer on C and beta
        self.optim.zero_grad() # We do this since it is possible that beta and C have gradients. 
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
        for j in tqdm(range(N_epoch)): 
            log_like = 0 # init the log likelihood. We will add each log likelihood of each batch to get the log likelihood 
                         # of the whole dataset. 
            for Y_b, covariates_b, O_b, selected_indices in self.get_batch(batch_size):
                #store the batches for a nicer implementation.
                self.Y_b, self.covariates_b, self.O_b = Y_b.to(device), covariates_b.to(device), O_b.to(device)
                self.selected_indices = selected_indices
                # compute the log likelihood of the batch and add it to log_likelihood of the whole dataset.
                # Note that we need to call this function in order to be able to call self.get_batch_grad_(C/beta)()
                log_like += self.infer_batch_p_theta(N_iter_mode, lr_mode).item() 
                batch_grad_C = - self.get_batch_grad_C()
                batch_grad_beta = - self.get_batch_grad_beta()
                # Given the gradients of the batch, we update the variance reducted gradient if needed. 
                # Note that we need to give the gradient of each sample in the batch, not the average gradient of the batch. 
                if vr is not None : 
                    vr.update_new_grad([batch_grad_beta,batch_grad_C], selected_indices)
                else: 
                    self.beta.grad = torch.mean(batch_grad_beta, axis = 0)
                    self.C.grad = torch.mean(batch_grad_C, axis = 0)
                
                self.optim.step() # optimize beta and C given the gradients.
                self.optim.zero_grad()
                self.keep_records() # keep track of some stat
            self.log_like = log_like/self.n*batch_size # the log likelihood of the whole dataset.
            self.average_likelihood() # average the  log likelihood for a criterion less random.
            crit = self.compute_criterion() # compute the criterion (i.e., check if the likelihood has increased) 
            
            # if the likelihood has not improved for self.nb_plateau iteration, we stop the algorithm.
            if crit > self.nb_plateau :
                print('Algorithm stopped after ', j, ' iterations')
                self.fitted = True
                break 
            # if we use SVRG VR, we need to compute the gradient of the whole dataset to update the average gradient. 
            if VR == 'SVRG':
                self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
                self.selected_indices = np.arange(0,self.n)
                self.get_gradient_requirement(N_iter_mode, lr_mode)
                full_grad_C = -self.get_batch_grad_C()
                full_grad_beta = -self.get_batch_grad_beta()
                vr.update_table([full_grad_beta,full_grad_C])
            # we lower the learning rate (that finds the mode) since after one epoch, we are not very far
            # from the new mode. Indeed, we found the mode for previous beta and C, and we now 
            # need to find the new mode for the new beta and C.We will begin our ascent from the previous mode found.  
            # However, since beta and C won't move very much,  
            # the mode for the new beta and C won't move very much either. We can lower the learning rate so that 
            # the optimizer will only adjust a little bit its starting position. 
            if j == 0 : 
                lr_mode/=10
            if j == 20 : 
                lr_mode/=3
            if j == 40 : 
                lr_mode/= 2
        # the model has been fitted
        self.fitted = True
        
    def compute_best_loglike(self, acc = 0.001, N_iter_mode = 100, lr_mode = 0.001):
        '''
        Estimate the best likelihood of the model, i.e. the likelihood estimated with beta_mean and C_mean. 
        
        args : 
            acc : float greater than 0. The accuracy you want for the estimation of the likelihood
            N_iter_mode : int. The number of iteration you are ready to do to find the mode. 
            lr_mode : float greater than 0. The learning of the gradient ascent you do to find the mode. 
        returns : 
            float. The estimated likelihood of beta_mean and C_mean
        '''
        self.Y_b, self.covariates_b, self.O_b = self.Y, self.covariates, self.O
        self.selected_indices = np.arange(0,self.n)
        # set beta and C as beta_mean and C_mean to compute the likelihood
        self.beta = torch.clone(self.beta_mean)
        self.C = torch.clone(self.C_mean)
        # infer p_theta
        self.best_log_like = self.infer_batch_p_theta(N_iter_mode, lr_mode, int(1/acc))
        return self.best_log_like
        
    def compute_criterion(self):
        '''
        Updates the criterion of the model. The criterion counts the number of times the likelihood has not improved. 
        We also append the criterion in a list in order to plot it after. 
        args :
            None 
        returns : the criterion
        '''
        if self.average_log_like > self.max_log_like : 
            self.max_log_like = self.average_log_like
        else : self.crit_cmpt +=1
        self.crit_cmpt_list.append(self.crit_cmpt)
        return self.crit_cmpt 

    
    def infer_batch_p_theta(self, N_iter_mode, lr_mode,take_mean = True): 
        '''
        Infer p_theta that is computed for a batch of the dataset. 
        args : 
            N_iter mode : int. The number of iteration you are ready to do to find the mode of the posterior
            lr_mode : float greater than 0. The learning rate of the gradient ascent that finds the mode.
            take_mean : bool. If we want the alogorithm to return the mean of the log likelihood or the log likelihood of each sample. 
        '''
        # get the gradient requirement. It also computes the weights of the IMPS
        self.get_gradient_requirement(N_iter_mode, lr_mode)
        # take the log of the weights and adjust with the missing constant self.const that has been removed before to avoid numerical 0. 
        log = torch.log(torch.mean(self.weights,axis = 0))+self.const#*self.mask
        
        # return the mean of the log likelihood of the batch if we want to, or the log of each sample in the batch.  
        if take_mean :
            return torch.mean(log)
        else: 
            return log 
    
    def get_gradient_requirement(self, N_iter_mode, lr_mode):
        '''
        does all the operation that we need to compute the gradients. 
        We need the gaussian samples and the weights, which we compute here. The gaussians samples needs to 
        be sampled from the right mean and variance, and we find this by calling find_batch_mode and get_batch_best_var methods.
        
        args : 
            N_iter_mode : int. The maximum number of iterations you are ready to do to find the mode. 
            lr_mode : float greater than 0. The learning rate of the gradient ascent that finds the mode.
        '''
        # get the mode
        self.find_batch_mode(N_iter_mode, lr_mode)
        # thanks to the mode, we can now get the best variance. 
        self.get_batch_best_var()
        # get the samples generated with the mean (mode) and variance found.  
        self.samples = sample_gaussians(self.N_samples, self.batch_mode, self.sqrt_Sigma_b)
        # get the weights
        self.weights = self.get_batch_weights()
        
    def get_batch_weights(self): 
        '''
        Compute the weights of the IMPS formula. Given the gaussian samples, the weights are computed as the ratio 
        of the likelihood of the posterior and the likelihood of the gaussian samples. Note that we first
        compute the logarithm of the likelihood of the posterior and the logarithm of the gaussian samples, then remove 
        the maximum to avoid numerical zero, and takes the exponential. We keep in memory the constant removed to get it back later. 
            args : None 
            
            returns : torch.tensor of size (N_samples,N_batch). The computed weights. 
        '''
        # log likelihood of the posterior
        self.log_f = self.batch_un_log_posterior(self.samples)
        # log likelihood of the gaussian density
        self.log_g = log_gaussian_density(self.samples, self.batch_mode, self.Sigma_b)
        # difference between the two logarithm
        diff_log = self.log_f-self.log_g 
        self.const = torch.max(diff_log, axis = 0)[0]
        # remove the maximum to avoid numerical zero. 
        diff_log -= torch.max(diff_log, axis = 0)[0]
        weights = torch.exp(diff_log)
        return weights
    def get_batch_best_var(self):
        '''
        Compute the best variance for the importance law. Given the mode, we can derive the best variance that fits the posterior. Why we do this is a little bit tricky, please see the doc to find out why we do so.
        args : None 
        returns : None but compute the best covariance matrix and its square root, stocked in the IMPS_PLN object. 
        '''
        batch_matrix = torch.matmul(self.C.unsqueeze(2), self.C.unsqueeze(1)).unsqueeze(0)
        CW = torch.matmul(self.C.unsqueeze(0),self.batch_mode.unsqueeze(2)).squeeze()
        common = torch.exp(self.O_b  + self.covariates_b@self.beta + CW).unsqueeze(2).unsqueeze(3)
        prod = batch_matrix*common
        # The hessian of the posterior
        Hess_post = torch.sum(prod, axis = 1)+torch.eye(self.q).to(device) 
        self.Sigma_b = torch.inverse(Hess_post.detach())
        #add a term to avoid non-invertible matrix. 
        eps = torch.diag(torch.full((self.q,1),1e-8).squeeze()).to(device)
        self.sqrt_Sigma_b = TLA.cholesky(self.Sigma_b+ eps)
        
    def get_batch_grad_beta(self): 
        '''
        Computes the gradient for beta for the batch. To see why we do so, please see the doc. 
        args : None
        returns : torch.tensor of size (batch_size,d,p). 
        '''
        first = torch.matmul(self.covariates_b.unsqueeze(2), self.Y_b.unsqueeze(1).double())
        XB = torch.matmul(self.covariates_b.unsqueeze(1), self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(self.C.reshape(1,1,self.p,1,self.q), self.samples.unsqueeze(2).unsqueeze(4)).squeeze()
        Xexp = torch.matmul(self.covariates_b.unsqueeze(0).unsqueeze(3), torch.exp(self.O_b + XB + CV).unsqueeze(2))
        sec = torch.sum(torch.multiply(self.weights.unsqueeze(2).unsqueeze(3), Xexp), axis = 0)/(torch.sum(self.weights, axis = 0).unsqueeze(1).unsqueeze(2))
        return first-sec
    
    def get_batch_grad_C(self): 
        '''
        Same as get_batch_grad_C but for C instead of beta. (returned size : (batch_size, p,q))
        ''' 
        XB = torch.matmul(self.covariates_b.unsqueeze(1), self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(self.C.reshape(1,1,self.p,1,self.q), self.samples.unsqueeze(2).unsqueeze(4)).squeeze()
        Ymoinsexp = self.Y_b - torch.exp(self.O_b + XB + CV)
        outer = torch.matmul(Ymoinsexp.unsqueeze(3), self.samples.unsqueeze(2))
        denum = torch.sum(self.weights, axis = 0)
        num = torch.multiply(self.weights.unsqueeze(2).unsqueeze(3), outer)
        batch_grad = torch.sum(num/(denum.unsqueeze(0).unsqueeze(2).unsqueeze(3)), axis = 0)
        return batch_grad
    
    
    def find_batch_mode(self, N_iter, lr, eps = 7e-3):
        '''
        Finds the mode of the posterior. As starting point, we will use the last mode computed. However, each mode depends on the batch (Y_b,O_b, covariates_b), so that we need to know which batch we are taking. That is why we stock the current batch took by stocking self.selected_indices to know which previous mode to take. 
        
        args : 
            N_iter : int. The maximum number of iteration you are ready to do to find the mode. 
            lr : float greater than 0. The learning of the optimizer for the gradient ascent. 
            eps : positive float. The tolerance. The algorithm will stop if the maximum of W_t-W_{t-1} is lower than eps.
        returns : 
            None, but compute and stock the mode in self.batch_mode and the starting point. 
        '''
        # The loss we will use for the gradient ascent. 
        def batch_un_log_posterior(W): 
            return batch_log_P_WgivenY(self.Y_b, self.covariates_b, self.O_b, W, self.C, self.beta) 
        self.batch_un_log_posterior = batch_un_log_posterior
        # get the corresponding starting point. 
        W = torch.clone(self.starting_point[self.selected_indices].detach()).to(device)
        W.requires_grad_(True)
        optimizer = torch.optim.Rprop([W], lr = lr)
        criterion = 2*eps
        old_W = torch.clone(W)
        i = 0
        keep_condition = True
        while  i < N_iter and keep_condition: 
            # compute the loss
            loss = -torch.mean(self.batch_un_log_posterior(W))
            # propagate the gradients
            loss.backward()
            # update the parameter
            optimizer.step()
            crit = torch.max(torch.abs(W-old_W))
            optimizer.zero_grad()
            if crit<eps and i > 2 : 
                keep_condition = False 
            old_W = torch.clone(W)
            i+= 1
        # keep the number of iteration as information 
        self.nb_iteration_list.append(i)
        # stock the mode
        self.batch_mode = torch.clone(W)
        # stock the starting point for the next epoch. 
        self.starting_point[self.selected_indices] = torch.clone(W)
        
    def Sigma(self): 
        '''
        Small method to get the Sigma of the model. 
        '''
        return (self.C_mean.detach())@(self.C_mean.detach().T)
    
    def show_Sigma(self): 
        '''
        Small method that displays Sigma
        '''
        sns.heatmap(self.Sigma())
        plt.show()
        
    def __str__(self):
        '''
        print the model, Sigma and the likelihood.
        '''
        try : 
            print('Log likelihood of the model : ', self.best_log_like) 
        except :  
            self.compute_best_loglike()
            print('Log likelihood of the model : ', self.best_log_like)     
        show(self)
        self.show_Sigma()
        return ''
    
   



