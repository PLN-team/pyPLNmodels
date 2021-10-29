import torch 
from torch.optim import Optimizer

if torch.cuda.is_available():
    device = torch.device('cuda')
else : 
    device = torch.device('cpu')

print('device :', device)
class SVRG(Optimizer):
    '''This class aims at implementing the SVRG optimizer.
    How to use it ? 
    First, declare it as you do in pytorch for a simple SGD optimizer. 
    Then, call optim.step() to do a gradient step. 
    When you want, update the gradient estimation calling 
    optim.update_mean_grads(grad_means). Do not do this too often, 
    since computing the gradien mean is very costly in general. 
    '''
    def __init__(self, params, lr):
        '''
        initialization for the SVRG optimizer. 
        
        args : 
            params : list with each parameters
            lr : learning rate 
        '''
            
        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)
    @torch.no_grad()            
    def step(self):
        """Performs a single optimization step.
        Args:
            batch_grads : list that contains the gradients computed for a subset 
            of the samples. should be a list of size nb_params and each item
            should be of size (batch_size, shape of the parameter)
            
            selected_indices : list of indices, the ones we used to estimate the gradient with.
            
        """
        for group in self.param_groups:
            params_with_grad = []
            lr = group['lr']
            for param in group['params']:
                if param.grad is not None:
                    params_with_grad.append(param)
            self.parameter_step(params_with_grad, lr)
    def parameter_step(self,params_with_grad, lr): 
        for i, param in enumerate(params_with_grad): 
            param-= lr*(param.grad-param.old_grad + param.grad_mean)
            
    def update_mean_grad(self, grad_means):
        for group in self.param_groups:
            for i,param in enumerate(group['params']): 
                param.grad_mean = grad_means[i]



class SAGA(Optimizer):
    '''
    This class aims at defining the SAGA optimizer in pytorch, deriving from 
    the torch.optim class. How tu use it ?
    First, you need to initialize the optimizer. It takes 3 arguments: the parameters, 
    the learning rates (as you do in pytorch) and the sample_size. 
    Then, you need to call optim.step(batch_grads, selected_indices). 
    batch_grads needs to be a list of gradients. It should be of length 
    (nb_parameter), and each element of this list should be of size (parameter.shape).
    Selected_indices is here to show wich gradient samples we need to update. 
    All the other indices will stay unchanged. 
    '''

    def __init__(self, params, lr, sample_size):
        '''
        initialization for the SAGA optimizer. 
        
        args : 
            params : list with each parameters
            lr : learning rate 
            sample_size : int. the number of samples you have in your dataset. 
        '''
            
        self.sample_size = sample_size
        # to know if we have already run through the dict. 
        self.run_through = False
        defaults = dict(lr=lr)
        super(SAGA, self).__init__(params, defaults)
        # initialize the number of sample we have for the gradient estimate.
        # the more iteration we will do, the more samples we will see, 
        # the more accurate our gradient estimate will be. 
        # note self.nb_non_zero won't be bigger than self.sample_size
        self.nb_non_zero = 0
        for group in self.param_groups:
            # for each parameter of the optimizer, we need to init 
            # the table (i.e. the quantities we use to estimate the gradients)
            # however, we need to keep every quantity since we will 
            # update some of them.
            for param in group['params']: 
                shape = list(param.shape)
                shape.insert(0,sample_size)
                # param.table is a tensor of size (N_samples, param.shape)
                param.table = torch.zeros(shape).to(device)
                # param.mean_table is the approximation of the full gradient, 
                # thus of size (param.shape)
                param.mean_table = torch.zeros(param.shape).to(device)
                
        
    
    def __setstate__(self, state):
        super(SAGA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    @torch.no_grad()
    def step(self, batch_grads, selected_indices):
        """Performs a single optimization step.
        Args:
            batch_grads : list that contains the gradients computed for a subset 
            of the samples. should be a list of size nb_params and each item
            should be of size (batch_size, shape of the parameter)
            
            selected_indices : list of indices, the ones we used to estimate the gradient with.
        """
        # we begin with storing some useful quantities.
        for group in self.param_groups:
            params_with_grad = []
            means_batch_table = []
            lr = group['lr']
            for i,param in enumerate(group['params']):
                if param.grad is not None:
                        params_with_grad.append(param)
                        means_batch_table.append(torch.mean(param.table[selected_indices], axis = 0))
            self.nb_non_zero = min(self.nb_non_zero + len(selected_indices),self.sample_size )
            self.parameter_step(params_with_grad, means_batch_table, lr)
            self.saga_table_step(params_with_grad, batch_grads,means_batch_table, selected_indices)
            # if the number of samples we use to estimate the gradient, it means
            # that we have run through all the dataset. 
            if self.nb_non_zero == self.sample_size : 
                self.run_through = True
    
    def parameter_step(self,params_with_grad,means_batch_table,lr):
        '''
        updates the parameters of the optimizer given the parameters, 
        their grads and the previous grads. 
        args : 
            params_with_grad : all the parameters of the optimizer that have gradients. 
            means_batch_table : the gradient of the previous iterate. 
            lr : learning rate used.   
        '''
        for i, param in enumerate(params_with_grad): 
            param-= lr*(param.grad-means_batch_table[i] + param.mean_table)
    
    

    def saga_table_step(self,params_with_grads, batch_grads,means_batch_table, selected_indices):
        '''
        let batch_size = len(selected_indices)
        update the table for each parameters. Given the selected_indices
        it will update for each parameter its table by replacing with the batch_size 
        new values the batch_size old values (the right ones, that's why 
        we need to give in parameter the selected_indices) , so 
        that (sample_size-batch_size) will be unchanged. 
        We also update the mean of the table for each parameter in an iterative manner
        (to avoid computing each time a mean of sample_size gradients, but only
        batch_size gradients)
        '''
        self.batch_size = len(selected_indices)
        for i, param in enumerate(params_with_grads): 
            # UPDATE OF THE MEAN_TABLE 
            # Note that the update rule of the table is different 
            # if we have not yet run through the whole dataset, so that we 
            # need to consider two cases. 
            if self.run_through == False : 
                param.mean_table *= (self.nb_non_zero-self.batch_size)/(self.nb_non_zero)
                param.mean_table += self.batch_size/(self.nb_non_zero)*(param.grad)
            else : 
                param.mean_table -= (self.batch_size/self.sample_size)*(means_batch_table[i]-param.grad)
            # UPDATE OF THE TABLE 
            param.table[selected_indices] = batch_grads[i]
            
class SAG(SAGA): 
    '''
    This class aims at defining the SAG optimizer. This class is the same as the SAGA 
    optimizer, except that it updats the parameter with a biased gradients. 
    '''
    def __init__(self, params, lr, sample_size):
        '''
        initialization for the SAGA optimizer. 
        
        args : 
            params : list with each parameters
            lr : learning rate 
            init_grad : for each param in params, for each sample. should be a list of length 
            nb_param. for each element in the list, the size should be 
            (N_samples, shape of the parameter)
            in the samples we have, we need a first gradient to store. 
            We will associate to each sample i the gradient of the corresponding function i  
            evaluated in the starting point. 
        '''
        super(SAG,self).__init__(params,lr, sample_size)
        
    def parameter_step(self,params_with_grad,means_batch_table,lr):
        for i, param in enumerate(params_with_grad): 
            # In SAGA, we only do not divise by self.sample_size the first argument 
            param-= lr*(1/self.sample_size*(param.grad-means_batch_table[i]) + param.mean_table)

            
   

            


        
        
        
        