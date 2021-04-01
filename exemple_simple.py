from gradient_descent import gradient_descent, minibatch_class
import utils
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from timeit import default_timer as timer

import numpy as np
import torch
from torch import nn
torch.set_default_dtype(torch.float64)

Y = read_csv('trichoptera.csv', sep=',').to_numpy()
O = np.outer(Y.sum(1), np.ones(Y.shape[1]))
data = utils.format_data(counts=Y, offsets=np.log(O))

# Class for PLN-PCA model


class model:
    def __init__(self, q, data, covariates=None, device='cpu'):
        self.Y = torch.tensor(data['Y'], device=device)

        # pre-compute an approximation of the log fact of the values of Y
        # those quantities will be used several times
        self.log_fact_Y = torch.tensor(
            utils.log_factorial_component_wise(self.Y), device=device)

        # offset
        self.O = torch.tensor(data['O'], device=device)

        self.n = data['Y'].shape[0]
        self.p = data['Y'].shape[1]
        self.q = q

        # covariates
        self.X = torch.ones((self.n, 1)) if covariates is None else torch.tensor(
            covariates, device=device)  # (i,l)
        self.d = self.X.shape[1]

        # dimensions
        self.dim = self.p*(self.d + self.q)

        # initialize parameters with zeros
        self.B = torch.tensor(
            np.zeros([self.p, self.d]), device=device)  # (j,l)
        self.C = torch.tensor(np.eye(self.p, self.q), device=device)  # (j,k)

        self.data = data

        self.device = device

        if self.X.shape[0] != self.n:
            raise NameError('Covariates and dataset size do not match')

        return None

    def xi(self, B, C, n_W_samples, minibatch=None):
        if minibatch is None:
            minibatch = list(range(0, self.n))
        Ws = torch.randn((n_W_samples, self.q), device=self.device)  # (W,k)
        # (i,j,W)
        return self.O[minibatch, :, None] + torch.tensordot(self.X[minibatch], B, dims=([1], [1]))[:, :, None] + torch.tensordot(C, Ws, dims=([1], [1]))[None, :, :], Ws

    def exp_arg(self, batch, xi):
        # (i,W)
        return torch.sum(self.Y[batch, :, None] * xi - torch.exp(xi) - self.log_fact_Y[batch, :, None], dim=1)

    def loglikelihood_estim(self, B, C, n_W_samples=None, minibatch=None):
        """computes the loglikelihood of the model for the given parameters or the current parameters of the model"""
        if n_W_samples is None:
            n_W_samples = 100*self.q

        # if minibatch is not specified, use whole dataset
        if minibatch is None:
            minibatch = list(range(0, self.n))

        xi, _ = self.xi(B, C, n_W_samples, minibatch)  # (i,j,W)

        exp_arg = self.exp_arg(minibatch, xi)

        foo = torch.exp(exp_arg)
        foo = torch.log(torch.mean(foo, dim=1))  # log(mean in W)
        foo = torch.mean(foo)  # mean in i

        return foo

    def loglikelihood_grad_estim(self, B=None, C=None, minibatch=None, n_W_samples=None):
        """computes gradient of the loglikelihood of the model for the given parameters or the current parameters of the model"""
        if minibatch is None:
            minibatch = list(range(0, self.n))
        if n_W_samples is None:
            n_W_samples = self.q

        Y = self.Y[minibatch, :].clone().detach()
        x = self.X[minibatch, :].clone().detach()

        xi, Ws = self.xi(B, C, n_W_samples, minibatch)

        exp_arg = self.exp_arg(minibatch, xi)  # (i,W)
        exp_arg = exp_arg + 50 - \
            torch.max(exp_arg, dim=1).values[:, None]  # (i,W)
        weights = torch.exp(exp_arg)  # (i,W)
        # renormalization
        weights = weights / torch.sum(weights, dim=1)[:, None]

        # (Y_ij - exp xi_ij(W)) * w_i(W)             # (i,j,W)
        weighted_values = (Y[:, :, None]-torch.exp(xi)) * weights[:, None, :]

        grad_estim_B = torch.mean(
            torch.sum(weighted_values[:, :, :, None] * x[:, None, None, :], dim=2), dim=0)
        grad_estim_C = torch.mean(
            torch.sum(weighted_values[:, :, :, None] * Ws[None, None], dim=2), dim=0)

        return [grad_estim_B, grad_estim_C]

    def fit_loglikelihood(self,
                          nsteps=100,
                          minibatch_size=30,
                          n_W_samples=None,
                          solver='RMSprop',
                          **kwargs):

        if minibatch_size is None:
            minibatch_size = self.n

        if n_W_samples is None:
            n_W_samples = 10*self.q

        minibatch = minibatch_class(self.n, minibatch_size)

        iterate, self.obj_values = gradient_descent([self.B, self.C],
                                                    gradient=self.loglikelihood_grad_estim,
                                                    objective=self.loglikelihood_estim,
                                                    nsteps=nsteps,
                                                    solver=solver,
                                                    minibatch=minibatch_class(
                                                        self.n, minibatch_size),
                                                    gradient_options={
                                                        'n_W_samples': n_W_samples},
                                                    device=self.device,
                                                    maximize=True,
                                                    **kwargs)
        self.B, self.C = iterate

        return None


foo = model(q=8, data=data)
foo.fit_loglikelihood(nsteps=1000, n_W_samples=500, minibatch_size=30, lr=.1)
