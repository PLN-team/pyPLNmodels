#!/usr/bin/env python

"""Implement a variational algorithm infering the parameter of the PLN model.
    We are only optimizing the ELBO (Evidence LOwer Bound), alternating between two steps:
        -M step: update the model parameters. We have a closed form for both parameters 
            for this step.
        -VE step: update the variational parameters. We do one step of gradient ascent to 
            update M and S. 
            
Created on Wed Nov  17 09:39:30 2021

@author: Bastien Batardiere, Julien Chiquet and Joon Kwon
"""

import math
import time 

import numpy as np
import torch
import seaborn as sns
import scipy.linalg as SLA 
torch.set_default_dtype(torch.float64)

#get the device
if torch.cuda.is_available(): 
    device = torch.device('cuda') 
else : 
    device = torch.device('cpu')
#device = torch.device('cpu')
print('device : ', device)


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
    tmp = torch.sum(  torch.multiply(Y, OplusM)  -torch.exp(OplusM+SrondS/2) +1/2*torch.log(SrondS))
    tmp -= 1/2*torch.trace(  
                            torch.mm(  
                                        torch.inverse(Sigma), 
                                        torch.diag(torch.sum(SrondS, dim = 0))+ torch.mm(MmoinsXB.T, MmoinsXB)
                                    )
                          )
    tmp-= n/2*torch.log(torch.det(Sigma))
    return tmp


class fastPLN():
    '''Implement the variational algorithm infering the parameters of the PLN model,
    with a closed form for the M step and a gradient step for the VE step. 
    '''
    def __init__(self): 
        '''Defines some usefuls lists and variables for the object. A deeper initalization is done 
        in the init_data() func, once the dataset is available.
        '''
        self.old_loss = 1
        # some lists to store some stats
        self.ELBO_list = list()
        self.running_times = list()
   
    def init_data(self,data): 
        '''Initialize the parameters with the right shape given the data. 
        
        Args: 
              data: list with 3 elements(torch.tensor): Y, O and covariates in this 
              order. Y and O should be of size (n,p), covariates of size (n,d). 
        Returns:
            None but initialize some useful data. 
        
        '''
        #known variables
        try : 
            self.Y = torch.from_numpy(data[0]).to(device);self.O = torch.from_numpy(data[1]).to(device);self.covariates =                   torch.from_numpy(data[2]).to(device)
        except : 
            self.Y = data[0].to(device);self.O = data[1].to(device);self.covariates = data[2].to(device)
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        
        #model parameter 
        noise = torch.randn(self.p) 
        self.Sigma =  (torch.diag(noise**2)+ 1e-1).to(device)
        self.beta = torch.rand((self.d, self.p)).to(device)
        #variational parameter
        self.M = torch.randn((self.n,self.p)).to(device)
        self.M.requires_grad_(True)
        self.S = torch.randn((self.n,self.p)).to(device)
        self.S.requires_grad_(True)
        
        self.params = {'S' : self.S,'M': self.M, 'beta' : self.beta, 'Sigma' : self.Sigma}
        
        
    ###################### parametrisation centered in X@\beta, variance CC.T ##############
    
    
    def compute_ELBO(self): 
        '''Compute the ELBO with the parameter of the model.'''
        return ELBO(self.Y,self.O , self.covariates,self.M ,self.S ,self.Sigma ,self.beta)
    
    
    def fit(self,Y,O,covariates, N_iter, tolerance = 0, optimizer = torch.optim.Rprop, lr = 0.7,verbose = False): 
        '''Main function of the class. Infer the best parameter Sigma and beta given the data.
        
        Args:
            Y: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            N_iter: int. The number of iteration you wnat to do.
            tolerance: non negative float. Criterion for the model (Default is 0). 
            optimizer: objects that inherits from torch.optim. The optimize you want. 
                Default is torch.optim.Rprop.
            lr: positive float. The learning rate of the optimizer. Default is 0.7
            verbose: bool. If True, will print some stats during the fitting. Default is False. 
            
        Returns: None but INITIALIZATION
        
        '''
        self.t0 = time.time()
        #initialize the data
        self.init_data([Y,O,covariates])
        self.optimizer = optimizer([self.S,self.M], lr = lr)
        stop_condition = False 
        i = 0
        while i < N_iter and stop_condition == False: 
            self.optimizer.zero_grad()
            loss = -self.compute_ELBO()
            loss.backward()
            self.optimizer.step()
            
            
            delta = self.old_loss - loss.item() # precision 
            # condition to see if we have reach the tolerance threshold
            if  abs(delta) < tolerance :
                stop_condition = True 
            self.old_loss = loss.item()
              
            self.ELBO_list.append(-loss.item())# keep track of the ELBO
            # print some stats if we want to
            if i%10 == 0 and verbose : 
                print('Iteration number: ', i)
                print('-------UPDATE-------')
                print('ELBO : ', np.round(-loss.item(),5))
                print('Delta : ', delta)
            i += 1
            #keep track of the time 
            self.running_times.append(time.time()-self.t0)
            #uupdate the parameters with their closed form. 
            self.beta = self.closed_beta()
            self.Sigma = self.closed_Sigma()
            
        # print some stats if we want to 
        if verbose : 
            if stop_condition : 
                print('---------------------------------Tolerance {} reached in {} iterations'.format(tolerance, i))
            else : 
                print('---------------------------------Maximum number of iterations reached : ', N_iter, 'last delta = ', delta)

        
    def closed_Sigma(self):
        '''
        closed form for Sigma with the first parametrisation centered in X\beta and variance Sigma 
        '''
        n,p = self.M.shape
        MmoinsXB = self.M-torch.mm(self.covariates,self.beta)
        return 1/(n)*(torch.mm(MmoinsXB.T,MmoinsXB) + torch.diag(torch.sum(torch.multiply(self.S,self.S), dim = 0)))
    def closed_beta(self): 
        '''
        closed form for beta with the first parametrisation above
        '''
        ## a améliorer l'inverse ! 
        return torch.mm(torch.mm(torch.inverse(torch.mm(self.covariates.T,self.covariates)), self.covariates.T),self.M)

    def show_Sigma(self):
        sns.heatmap(self.Sigma.detach().numpy())









