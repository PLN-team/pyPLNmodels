#!/usr/bin/env python

"""Implement 3 variance reduction algorithms (i.e. algorithms
approximating the gradient of a loss composed of a mean of n functions), namely:
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
else:
    device = torch.device('cpu')


class SAGARAD():
    '''Aims at computing the effective gradients of the SAGA algorithm. For example,
    given a gradient computed on a batch of our dataset, SAGA 'corrects' the gradients to
    try to approximate the gradient of the whole dataset, doing the following:
    SAGA will store the gradient with respect to each parameter for each sample in your
    dataset. For example, if your dataset has 200 samples and two parameter (of size d1 and d2)
    to optimize, then, it will store a list of two torch.tensor. The first will have size
    (200, d1) and the second (200, d2). Each of those tensors is here to approximate the
    gradient of the function (instead of taking only the gradient estimating on
    mini-batches, we try to estimate it taking the gradient of all the samples).
    In the following this list of torch.tensors will be called 'table'. At each iteration,
    it requires the gradient for each sample. For example, if batch_size = 10, we require a
    torch.tensor of size (10, parameter.shape), and we will replace those gradients
    in the table. To see how we update the gradient, please see the article.

    How tu use it?

    First, you need to declare it by calling for example (if you have only two params)
    'sagarad = SAGARAD([first_param, second_param], sample_size)'
    where sample size is the the number of samples you have in your dataset. Then,
    given a batch, you neeed to give him the gradient for each sample in your batch,
    for each parameter. To do so, just call
    'sagrad.update_new_grad([gradients for the first param,
                            gradients for the second param], selected_indices)'
    where selected_indices is the indices you selected for your batch,
    i.e. Y_batch = Y[selected_indices]. Each parameter does not require to
    have a gradient (i.e. param.grad can be None before calling
    self.update_new_grad()), but it will have a gradient after calling this method.
    The resulting gradient of the parameter will therefore be the variance reducted
    gradient.
    '''

    def __init__(self, params, sample_size):
        '''Define some usefuls attributes of the object, such as the parameters
        and the sample size. Need the sample size in order to initialize the
        table (this large vector will be needed to average the gradient.)

        Args:
            params: list. Each element of the list should be a torch.tensor object.
            sample_size: int. The number of sample in your dataset.
        Returns:
            None
        '''
        self.params = params
        self.sample_size = sample_size
        self.nb_non_zero = 0
        self.run_through = False
        self.bias = 1
        for param in params:
            shape = list(param.shape)
            shape.insert(0, sample_size)
            # Initialization of the table for each param with zeros.
            param.table = torch.zeros(shape).to(device)
            param.mean_table = torch.zeros(param.shape).to(device)

    def update_new_grad(self, batch_grads, selected_indices):
        '''Update the gradient of each parameter of the object with the SAGA formula.
        Note that it only needs to change the bias to get the SAG formula.

        Args:
            batch_grads: list of torch.tensors objects. Each object should be of size
                (batch_size, parameter.shape). Note that the input list should match
                the input list in the initialization. i.e., if the list in the
                __init__ begins with the parameter beta, this list should begins with
                the corresponding gradients for parameter beta.
            selected_indices: list of size batch_size. The indices of your batch. We
                need this to store the new gradients in the table. If Y is your dataset,
                then Y_batch should be equal to Y[selected_indices].
        Returns:
            None but updates the gradient of each parameter with the variance reducted
            gradient.
        '''
        means_batch_table = []
        self.batch_size = len(selected_indices)
        # Number of samples already seen.
        self.nb_non_zero = min(
            self.nb_non_zero +
            self.batch_size,
            self.sample_size)

        for i, param in enumerate(self.params):
            means_batch_table = torch.mean(
                param.table[selected_indices], axis=0)
            # Gradient formula in the SAGA optimizer
            batch_grad = torch.mean(batch_grads[i], axis=0)
            param.grad = (self.bias * (batch_grad -
                          means_batch_table) + param.mean_table)
            # Update the table with the new gradients
            if self.run_through == False:
                param.mean_table *= (self.nb_non_zero -
                                     self.batch_size) / (self.nb_non_zero)
                param.mean_table += (self.batch_size /
                                     (self.nb_non_zero) *
                                     (batch_grad)).detach()
            else:
                param.mean_table -= ((self.batch_size / self.sample_size)
                                     * (means_batch_table - batch_grad)).detach()
            # UPDATE OF THE TABLE
            param.table[selected_indices] = batch_grads[i].detach()
        if self.nb_non_zero == self.sample_size:
            self.run_through = True


class SAGRAD(SAGARAD):
    '''SAGRAD algorithm. Same as SAGARAD, only the biais in the update rule
    is changed, so that it only inherit the SAGARAD class and changes only the bias.
    '''

    def __init__(self, params, sample_size):
        '''Calls the initialization of the SAGARAD class, and only changes the bias.'''
        super(SAGRAD, self).__init__(params, sample_size)
        self.bias = 1 / self.sample_size


class SVRGRAD():
    '''SVRGRAD class. Implement the SVRG algorithm. Is very similar to the SAGRAD class.'''

    def __init__(self, params, sample_size):
        '''Define some useful attributes, such as the sample size and the parameters.
        Args:
            params. list of torch.tensors objects.
            sample_size: int. The number of sample in your dataset.
        Returns:
            a SVRGRAD object.
        '''
        self.sample_size = sample_size
        self.params = params
        for param in params:
            shape = list(param.shape)
            shape.insert(0, sample_size)
            param.table = torch.zeros(shape).to(device)
            param.mean_table = torch.zeros(param.shape).to(device)

    def update_new_grad(self, batch_grads, selected_indices):
        '''Update the gradient of each parameter of the object with the SVRG formula.
        Args:
            batch_grads: list of torch.tensors objects. Each object should be of size
                (batch_size, parameter.shape). Note that the input list should match
                the input list in the initialization. i.e., if the list in the
                __init__ begins with the parameter beta, this list should begins with
                the corresponding gradients for parameter beta.
            selected_indices: list of size batch_size. The indices of your batch. We
                need this to know which gradients to take in the table for the SVRG formula.
        Returns:
            None but update the gradient of each parameter with the variance
                reducted gradient.
        '''
        means_batch_table = []
        self.batch_size = len(selected_indices)

        for i, param in enumerate(self.params):
            batch_grad = torch.mean(batch_grads[i], axis=0)
            means_batch_table = torch.mean(
                param.table[selected_indices], axis=0).detach()
            # gradient formula in the SVRG optimizer
            param.grad = (
                batch_grad -
                means_batch_table +
                param.mean_table).detach()

    def update_table(self, new_tables):
        '''Update the table with a new table. If you have 2 parameters and 1000
        samples in your dataset, new table should be a list of 2 torch.tensor
        elements, with shape (1000, parameter1.shape), (1000, parameter2.shape).

        Args:
            new_tables: list of torch.tensor elements. Each element should
                be the new table of the corresponding parameter.
        Returns:
            None but update the mean of the table and the table of each parameter.
        '''
        for i, param in enumerate(self.params):
            param.table = new_tables[i].detach()
            param.mean_table = torch.mean(new_tables[i], axis=0).detach()
