#!/usr/bin/env python

"""Implemented 3 variance reduction algorithms (i.e. algorithms that 
approximate the gradient of a loss composed of a mean of n functions), namely: 
    - SAGA,   see Defazio, Aaron et al. “SAGA: A Fast Incremental Gradient
        Method With Support for Non-Strongly Convex Composite Objectives.” 
        for more details. 
    - Stochastic Average Gradient (SAG), see Schmidt, Mark W. et al. “Minimizing finite sums with the 
        stochastic average gradient.” Mathematical Programming 162 (2017): 83-112.
        for more details
    - SVRG, see Rie Johnson and Tong Zhang. "Accelerating stochastic 
    gradient descent using predictive variance reduction." NIPS, 2013 for more details.
    

Created on Wed Nov  17 09:39:30 2021

@author: Bastien Batardiere, Julien Chiquet and Joon Kwon
"""
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')
#device = torch.device('cpu') # have to deal with this. 



class SAGARAD():
    '''Aims at computing the effective gradients of the SAGA algorithm. For example, 
    given a gradient computed on a batch of our dataset, SAGA 'corrects' the gradients to 
    try to approximate the gradient of the whole dataset, doing the following: 
    SAGA will store the gradient with respect to each parameter for each sample in your 
    dataset. For example, if your dataset has 200 samples and two parameter to optimize, 
    then, it will store a list of two torch.tensor of size (200, parameter1.shape) 
    and (200, parameter2.shape). Each of those tensors is here to approximate the 
    gradient of the function (instead of taking only the gradient estimating on 
    mini-batches, we try to estimate it taking the gradient of all the samples). 
    We call in the following this list of torch.tensors 'table'.  
    
    takes the gradient computed on this batch, removing an old gradient, and adding 
    the old approximation of the gradient of the whole dataset, and then update the 
    old approximation. 
    
    How tu use it? 
    
    First, you need to declare it by calling for example (if you have only two params)
    'sagarad = SAGARAD([first_param, second_param], sample_size)'
    where sample size is the size of your dataset. Then, given a batch, you neeed to
    give him the gradient for each sample in your batch, for each parameter. To do so, 
    just call 
    'sagrad.update_new_grad([gradients for the first param, 
                            gradients for the second param], selected_indices)'
    where selected_indices is the indices you selected for your batch, 
    i.e. Y_batch = Y[selected_indices]. Each parameter does not require to 
    have a gradient (i.e. param.grad can be None before calling 
    self.update_new_grad()), but it will have a gradient after calling this function.
    The resulting gradient of the parameter will therefore be the variance reducted
    gradient.
    
    '''
    
    
    def __init__(self,params,sample_size):
        '''Defines some usefuls attributes of the object, such as the parameters 
        and the sample size. We need the sample size in order to initialize the 
        gradient of each sample (this large vector will be needed to average the gradient.)
        
        Args: 
            params: list. Each element of the list should be a torch.tensor object.
                
        '''
        
        self.params = params
        self.sample_size = sample_size
        self.nb_non_zero = 0
        self.run_through = False 
        self.bias = 1
        for param in params:
            shape = list(param.shape)
            shape.insert(0,sample_size)
            param.table = torch.zeros(shape).to(device)
            param.mean_table = torch.zeros(param.shape).to(device)
            
    def update_new_grad(self, batch_grads, selected_indices): 
        means_batch_table = []
        self.batch_size = len(selected_indices)
        self.nb_non_zero = min(self.nb_non_zero + self.batch_size,self.sample_size)

        for i,param in enumerate(self.params): 
            means_batch_table = torch.mean(param.table[selected_indices], axis = 0)
            # gradient formula in the SAGA optimizer
            batch_grad = torch.mean(batch_grads[i], axis = 0)
            param.grad = (self.bias*(batch_grad-means_batch_table) + param.mean_table)
            ## update the table with the new gradients we just got
            if self.run_through == False : 
                param.mean_table *= (self.nb_non_zero-self.batch_size)/(self.nb_non_zero)
                param.mean_table += (self.batch_size/(self.nb_non_zero)*(batch_grad)).detach()
            else : 
                param.mean_table -= ((self.batch_size/self.sample_size)*(means_batch_table-batch_grad)).detach()
            # UPDATE OF THE TABLE 
            param.table[selected_indices] = batch_grads[i].detach()
        if self.nb_non_zero == self.sample_size : 
            self.run_through = True
            
class SAGRAD(SAGARAD):
    def __init__(self, params, sample_size):
        super(SAGRAD, self).__init__(params, sample_size)
        self.biais = 1/self.sample_size
        
        
class SVRGRAD():
    def __init__(self, params,sample_size):
        self.sample_size = sample_size
        self.params = params
        for param in params:
            shape = list(param.shape)
            shape.insert(0,sample_size)
            param.table = torch.zeros(shape).to(device)
            param.mean_table = torch.zeros(param.shape).to(device)

    
    def update_new_grad(self,batch_grads, selected_indices):
        '''
        update the gradients of each parameter with the formula given in the 
        SVRG. 
        '''
        means_batch_table = []
        self.batch_size = len(selected_indices)

        for i,param in enumerate(self.params): 
            batch_grad = torch.mean(batch_grads[i], axis = 0)
            means_batch_table = torch.mean(param.table[selected_indices], axis = 0).detach()
            # gradient formula in the SAGA optimizer
            param.grad = (batch_grad-means_batch_table + param.mean_table).detach()
    def update_table(self, new_tables): 
        for i,param in enumerate(self.params): 
            param.table = new_tables[i].detach()
            param.mean_table = torch.mean(new_tables[i], axis = 0).detach()