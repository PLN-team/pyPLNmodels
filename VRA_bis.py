# This file aims at building an object that will compute 
# the gradients iteration of the SAGA, SAG and SVRG algorithm. 
# We do so in order to give those gradients to an optimizer like RMSprop so that it goes faster.
# since we do not have momentum or anything like this in the SAGA, SAG and SVRG optimizers. 
# VRA for variance reductor algorithm
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')



class SAGARAD(): 
    '''
    class that aims at computing the effective gradients of the SAGA algorithm. For example, 
    given a gradient computed on a batch of our dataset, SAGA 'correct' the gradients to 
    try to approximate the gradient of the whole dataset, doing the following : 
    takes the gradient computed on this batch, removing an old gradient, and adding 
    the old approximation of the gradient of the whole dataset. 
    
    How tu use it ? 
    
    First, you need to declare it by calling for example 
    sagarad = SAGARAD(list(first_param, second_param], sample_size)
    where sample size is the size of your dataset. 
    then, given a batch, you neeed to give him the gradient for each sample in 
    your batch, for each parameter. To do so, just call
    sagrad.update_new_grad([gradients for the first param, gradients for the second param], selected_indices)
    where selected_indices is the indices you selected for your batch, 
    i.e. Y_batch = Y[selected_indices]
    Then, it will update the gradient on his own. 
    each parameter does not require to have a gradient (i.e. param.grad can be None before
    calling self.update_new_grad()), but it will have a gradient after calling this function. 
    
    '''
    
    
    def __init__(self,params,sample_size, tables):
        self.params = params
        self.sample_size = sample_size
        self.bias = 1
        for i,param in enumerate(params):
            param.table = tables[i].to(device)
            param.mean_table = torch.mean(param.table, axis = 0)
            
    def update_new_grad(self, batch_grads, selected_indices): 
        means_batch_table = []
        self.batch_size = len(selected_indices)

        for i,param in enumerate(self.params): 
            means_batch_table = torch.mean(param.table[selected_indices], axis = 0)
            # gradient formula in the SAGA optimizer
            batch_grad = torch.mean(batch_grads[i], axis = 0)
            param.grad = (self.bias*(batch_grad-means_batch_table) + param.mean_table)
            ## update the table with the new gradients we just got
            param.mean_table -= ((self.batch_size/self.sample_size)*(means_batch_table-batch_grad)).detach()
            # UPDATE OF THE TABLE 
            param.table[selected_indices] = batch_grads[i].detach()
            
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

    
    def update_new_grad(self, selected_indices):
        '''
        update the gradients of each parameter with the formula given in the 
        SVRG. Note that each parameter must already have a gradient, we will 
        only update it (remove the old gradients and add the mean of the table)
        '''
        means_batch_table = []
        self.batch_size = len(selected_indices)

        for i,param in enumerate(self.params): 
            means_batch_table = torch.mean(param.table[selected_indices], axis = 0).detach()
            # gradient formula in the SAGA optimizer
            param.grad = (param.grad-means_batch_table + param.mean_table).detach()
    def update_table(self, new_tables): 
        for i,param in enumerate(self.params): 
            param.table = new_tables[i].detach()
            param.mean_table = torch.mean(new_tables[i], axis = 0).detach()