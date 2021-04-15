import torch
import numpy as np
import random
import math
from scipy.optimize import minimize
from itertools import zip_longest


class minibatch_class:
    """ Give succesive minibatches to use in the form of list of indices"""

    def __init__(self, dataset_size, minibatch_size=None, method='linear'):
        self.dataset_size = dataset_size
        self.batch = list(range(0, dataset_size))
        random.shuffle(self.batch)
        self.minibatch_size = minibatch_size
        self.cursor = 0
        self.method = method
        return None

    def get(self, minibatch_size=None):
        size = self.minibatch_size if minibatch_size is None else minibatch_size
        if self.method == 'linear':
            upper_bound = min(self.cursor+size, self.dataset_size)
            remaining = size-(upper_bound-self.cursor)
            minibatch = self.batch[0:remaining] + \
                self.batch[self.cursor:upper_bound]
            self.cursor = (self.cursor + size) % self.dataset_size
            return minibatch
        elif self.method == 'resample':
            return random.sample(self.batch, size)
        elif self.method == 'bootstrap':
            return random.choices(self.batch, size)


class adaptive_step_sizes:
    def __init__(self, shape, method='RMSprop', lr=.5, lr_scheduler=None,
                 second_moment_decay=None, eps=10**(-8), device='cpu'):
        """
        lr_scheduler, if given, must be a function which gives a coefficient as a function of the time t. This coeffcient will be applied to the learning rate. By default, constant coefficient 1 will be applied.
        """

        self.G = torch.zeros(shape, device=device)
        self.method = method
        self.t = 0

        if method.lower() == 'adadelta':
            self.lr = 1
        else:
            self.lr = lr

        if method.lower() == 'adamax':
            self.eps = 0
        else:
            self.eps = eps

        if second_moment_decay is None:
            # Default value for second_moment_decay depends on the method
            if method.lower() == 'rmsprop':
                self.second_moment_decay = .9
            elif method.lower() == 'adadelta':
                self.second_moment_decay = .9
            elif method.lower() in ['adam', 'adamax']:
                self.second_moment_decay = .999
        else:
            self.second_moment_decay = second_moment_decay

        if ((lr_scheduler is None) or method.lower() == 'adadelta'):
            # By default, lr_scheduler gives constant coefficient 1
            self.lr_scheduler = lambda t: 1
        else:
            self.lr_scheduler = lr_scheduler

        if method.lower() == 'sgd':
            shape = 1
            self.G = torch.ones(1)

        return None

    def update(self, grad_estim):
        self.t += 1
        if self.method.lower() in ['rmsprop', 'adam', 'adadelta']:
            self.G = self.second_moment_decay*self.G + \
                (1-self.second_moment_decay)*torch.mul(grad_estim, grad_estim)
        elif self.method.lower() == 'adagrad':
            self.G += torch.mul(grad_estim, grad_estim)
        elif self.method.lower() == 'adamax':
            self.G = torch.max(self.second_moment_decay*self.G,
                               torch.abs(grad_estim))
        return self

    def get_step_sizes(self):
        if self.method.lower() == 'adam':
            # compute bias-corrected second moment estimate
            denominator = torch.sqrt(
                self.G/(1 - self.second_moment_decay**self.t)) + self.eps
        elif self.method.lower() == 'adamax':
            denominator = self.G
        else:  # works for sgd, rmsprop, adagrad, adadelta
            denominator = torch.sqrt(self.eps + self.G)

        return self.lr * self.lr_scheduler(self.t) * torch.reciprocal(denominator)


def gradient_descent(initial_point,
                     # a function which computes a gradient (estimate) at a given point
                     gradient,
                     nsteps,  # number of steps to be performed
                     # algorithm to be used (sgd, rmsprop, adam, adamax, adadelta, etc.)
                     solver,
                     lr,  # learning rate
                     objective,  # a function which computes the value of the objective function at a given point
                     lr_scheduler=None,  # a function which gives a time - varying for the learning rate
                     momentum=None,  # option used by some algorithms
                     second_moment_decay=None,  # option used by some algorithms
                     weight_decay=0,  # option used by some algorithms
                     minibatch=False,  # pass the minibatch object
                     # additional arguments to be passed to the function computing the gradient
                     gradient_options={},
                     # additional arguments to be passed to the function computing the objective
                     objective_options={},
                     maximize=False,  # whether the objective function is to be maximized or minimized
                     device='cpu',  # pytorch option regarding the device on which the computation must be performed
                     **kwargs):
    """Performs a gradient descent. `initial_point` can either be a single tensor, or a list of tensors which then represents several blocks of variables.  Implemented algorithms are: SGD, Adagrad, RMSprop, Adam, Adamax, Adadelta.  Implemented options are: lr schedule,  weight decay."""
    if solver.lower() in ['adam', 'adamax']:
        momentum = .9
    else:
        momentum = 0

    if second_moment_decay is None:
        if solver.lower() == 'adadelta':
            second_moment_decay = .9

    # If initial_point is given as a tensor, turn it into a list containing that single tensor
    if not isinstance(initial_point, list):
        initial_point = [initial_point]

    iterate = []
    step_sizes = []

    # number of tensors (corresponding to the number of )
    n_block_variables = len(initial_point)

    # store initial point
    for i in range(n_block_variables):
        iterate.append(initial_point[i])
        step_sizes.append(adaptive_step_sizes(
            method=solver, shape=tuple(initial_point[i].shape),
            lr=lr, lr_scheduler=lr_scheduler,
            second_moment_decay=second_moment_decay))

    obj_values = None if objective in [None, False] else []

    first_order_estimate = []
    delta_running_average = []
    for i in range(n_block_variables):
        first_order_estimate.append(
            torch.zeros_like(initial_point[i], device=device))

        if solver.lower() == 'adadelta':
            delta_running_average.append(
                torch.zeros_like(initial_point[i], device=device))

    for t in range(0, nsteps):
        if minibatch is not False:
            minibatch_ = minibatch.get()
            gradient_options['minibatch'] = minibatch_

        # compute the gradient and objective function value at the current iterate
        iterate = iterate

        grad = gradient(*iterate, **gradient_options)

        value = None if obj_values is None else objective(
            *iterate, **objective_options)

        if not isinstance(grad, list):
            grad = [grad]

        # store the value of the value of the objective function
        if obj_values is not None:
            obj_values.append(value)

        for i in range(n_block_variables):
            # weight decay
            if weight_decay != 0:
                if maximize:
                    grad[i] -= weight_decay * iterate[i]
                else:
                    grad[i] += weight_decay * iterate[i]

            # update the step sizes
            step_sizes[i].update(grad[i])

            # update first order estimate
            first_order_estimate[i] = momentum * \
                first_order_estimate[i] + (1 - momentum)*grad[i]

            # gradient step
            step = torch.mul(
                step_sizes[i].get_step_sizes(), first_order_estimate[i])

            if solver.lower() in ['adam', 'adamax']:
                # bias correction
                step /= (1 - momentum**(t + 1))
            elif solver.lower() == 'adadelta':
                step *= torch.sqrt(delta_running_average[i] + 10**(-7))
                delta_running_average[i] = second_moment_decay * \
                    delta_running_average[i] + \
                    (1 - second_moment_decay)*(step**2)

            if maximize is False:
                step[i] *= -1

            iterate[i] = iterate[i] + step

    return (iterate, obj_values)
