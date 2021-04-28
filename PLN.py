#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:07:44 2019

@author: jchiquet
"""

import numpy as np
from scipy import linalg
from scipy import optimize
import nlopt
import torch

from torch import nn as nn

## my own functions
import utils

# Class for PLN model
class model:
    
     ## Constructor
    def __init__(self, p, d) :
        self.Sigma  = np.eye(p)
        self.B      = np.zeros([d,p])
        self._p     = p
        self._d     = d

class full_covariance(model):
    
    def __init__(self, data) :
        
        #initializing the data
        self._data = data
        self._n    = data['Y'].shape[0]
        
        # initializing the model
        super(full_covariance, self).__init__(self._data['Y'].shape[1], self._data['X'].shape[1])

    def profiled_variational_objective(self, M, S, mu, Sigma) :
    
        Omega = linalg.inv(Sigma)
        S2 = S * S
        A = np.exp(self._data['O'] + M + .5 * S2)
    
        # A. terms of the objective function
        # log p(Y|Z) integrated over E_q, with q the variational distribution
        E_log_p_Y_Z = np.sum(self._data['Y'] * (self._data['O'] + M) - A) - self._data['KY']
     
        # log p(Z) integrated over E_q, with q the variational distribution
        # with plugged estimator of Sigma and mu
        E_log_p_Z = -.5 * (self._p + self._n * np.linalg.slogdet(Sigma)[1] - self._n * self._p)
        
        # Entropy of the variational distribution q
        H_q_Z = .5 * np.sum(np.log(S2)) 
    
        # B. terms of the gradient
        grd_M = (M - mu) @ Omega - self._data['Y'] +  A
        grd_S = S * (np.outer(np.ones(self._n), np.diag(Omega)) + S * A - 1/S)    
      
        return -(E_log_p_Y_Z + E_log_p_Z + H_q_Z), utils._variational_mat2vec(grd_M, grd_S)

    def variational_objective(self, M, S, mu, Sigma) :
    
        Omega = linalg.inv(Sigma)
        S2 = S * S
        A = np.exp(self._data['O'] + M + .5 * S2)
    
        # A. terms of the objective function
        # log p(Y|Z) integrated over E_q, with q the variational distribution
        E_log_p_Y_Z = np.sum(self._data['Y'] * (self._data['O'] + M) - A) - self._data['KY']
     
        # log p(Z) integrated over E_q, with q the variational distribution
        # with plugged estimator of Sigma and mu
        E_log_p_Z = -.5 * (np.trace(Omega @ (M - mu).T @ (M - mu)) + sum(np.diag(Omega) * np.sum(S2, 0)) + self._n * np.linalg.slogdet(Sigma)[1] - self._n * self._p)
        
        # Entropy of the variational distribution q
        H_q_Z = .5 * np.sum(np.log(S2)) 
    
        # B. terms of the gradient
        grd_M = (M - mu) @ Omega - self._data['Y'] +  A
        grd_S = S * ( (np.outer(np.ones(self._n), np.diag(Omega)) + A) - 1/S)
      
        return -(E_log_p_Y_Z + E_log_p_Z + H_q_Z), utils._variational_mat2vec(grd_M, grd_S)

    def optim_profiled_variational_nlopt(self, x0, xtol, ftol):
        
        def loss_and_grad(x, grad):
            M, S      = utils._variational_vec2mat(x, self._n, self._p)
            mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
            objective, grad[:] = self.variational_objective(M, S, mu, Sigma)        
            return objective
        
        optimizer = nlopt.opt(nlopt.LD_CCSAQ, x0.size)
        optimizer.set_xtol_rel(xtol)
        optimizer.set_ftol_abs(0)
        optimizer.set_ftol_rel(ftol)
        optimizer.set_min_objective(loss_and_grad)
        x = optimizer.optimize(x0)
        M, S      = utils._variational_vec2mat(x, self._n, self._p)
        return optimizer.last_optimum_value(), M, S

    def optim_profiled_variational_scipy(self, x0, ftol):
        
        def loss_and_grad(x):
            M, S      = utils._variational_vec2mat(x, self._n, self._p)
            mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
            return self.profiled_variational_objective(M, S, mu, Sigma)
       
        optimizer = optimize.minimize(
            fun     = loss_and_grad,
            x0      = x0,
            jac     = True,
            method  = 'L-BFGS-B')
            #options = {'ftol': ftol})
        M, S = utils._variational_vec2mat(optimizer.x, self._n, self._p)
        return optimizer.fun, M, S

    def optim_profiled_variational_pytorch(self, M0, S0, nsteps, lr, ftol, device):
        
        M = nn.Parameter(torch.tensor(M0, requires_grad = True, device = device))
        S = nn.Parameter(torch.tensor(S0, requires_grad = True, device = device))
        
        projX = torch.tensor(self._data['projX'], requires_grad = False, device = device)
        O = torch.tensor(self._data['O'], requires_grad = False, device = device)
        Y = torch.tensor(self._data['Y'], requires_grad = False, device = device)

        def loss_torch(M, S) :

            # replace mu and Sigma per their explicit estimator
            mu = projX @ M
            
            # replace mu and Sigma per their explicit estimator
            Sigma = (M - mu).T @ (M - mu) / self._n + torch.diag((S*S).mean(0))
      
            # A. terms of the objective function
            # log p(Y|Z) integrated over E_q, with q the variational distribution
            E_log_p_Y_Z = torch.sum(Y * (M + O) - torch.exp(O + M + .5 * S*S)) - self._data['KY']
         
            # log p(Z) integrated over E_q, with q the variational distribution
            # with plugged estimator of Sigma and mu
            E_log_p_Z = -.5 * (self._p + self._n * torch.logdet(Sigma) + self._n * self._p)
            
            # Entropy of the variational distribution q
            H_q_Z = .5 * torch.sum(torch.log(S*S))
          
            return -(E_log_p_Y_Z + E_log_p_Z + H_q_Z)
           
#       optimizer = torch.optim.RMSprop([M, S], lr = lr, eps = 1e-2)
        optimizer = torch.optim.Rprop([M, S], lr = lr)

        objective = np.ndarray([nsteps + 1, 1])
        cond = False
        iterate = 0
        delta = 2 * ftol
        while cond == False:
            iterate += 1
            optimizer.zero_grad()
            loss = loss_torch(M, S)
            loss.backward()
            
            # Access gradient if necessary
            optimizer.step()
            if (iterate > nsteps or delta < ftol): 
                cond = True
            else:
                objective[iterate] = loss.data.cpu().numpy()
                delta = np.abs(objective[iterate] - objective[iterate-1])/np.abs(objective[iterate])
                # print('Step # {}, loss: {}, delta: {}'.format(iterate, loss.item(), delta))            
           
        M = M.data.cpu().numpy()
        S = S.data.cpu().numpy()
        return objective[0:iterate, 0], M, S
        
    def fit_profiled_variational_objective(self, M0, S0, solver = 'nlopt', ftol = 1e-12, xtol = 1e-6, nsteps = int(5e3), lr = 0.1, use_hessian = 'false', device = 'cpu'):
        
        x0 = np.concatenate((M0.flatten(), np.sqrt(S0.flatten())))
    
        if (solver == 'scipy'):
            [objective, M, S] = self.optim_profiled_variational_scipy(x0, ftol)
         
        if (solver == 'nlopt'):
            [objective, M, S] = self.optim_profiled_variational_nlopt(x0, xtol, ftol)

        if (solver == 'torch'):
            [objective, M, S] = self.optim_profiled_variational_pytorch(M0, S0, nsteps, lr, ftol, device)

        # replace mu and Sigma per their explicit estimator
        mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
        self.B = linalg.inv(self._data['X'].T @ self._data['X']) @ self._data['X'].T @ M
        self.Sigma = Sigma

        vloglik, gradient = self.profiled_variational_objective(M, S, mu, self.Sigma)
        results = {'means' : M, 'variances' : S * S, 'criterion' : vloglik, 'gradient' : gradient, 'objective': objective}
        return results        

    def fit_variational_objective(self, M, S, solver = 'ccsa', ftol = 1e-12, xtol = 1e-8, nsteps = int(1e4), lr = 0.05, use_hessian = 'false'):

        mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
        
        cond = False
        iterate = 0
        objective = np.ndarray([nsteps, 1])
        
        ## initialize the NLopt optimzer
        optimizer = nlopt.opt(nlopt.LD_CCSAQ, 2 * self._n * self._p)
        optimizer.set_xtol_rel(xtol)
        optimizer.set_ftol_abs(0)
        optimizer.set_ftol_rel(ftol)

        while cond == False:                                

            # E step
            def loss_and_grad(x, grad):
                M, S      = utils._variational_vec2mat(x, self._n, self._p)
                objective, grad[:] = self.variational_objective(M, S, mu, Sigma)        
                return objective

            optimizer.set_min_objective(loss_and_grad)
            x0 = np.concatenate((M.flatten(), S.flatten()))
            x = optimizer.optimize(x0)

            # M step
            M, S      = utils._variational_vec2mat(x, self._n, self._p)
            mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])

            ## assessing convergence            
            objective[iterate]=  optimizer.last_optimum_value()
            delta = np.abs(objective[iterate] - objective[iterate-1])/np.abs(objective[iterate])
            print('Step # {}, loss: {}, delta: {}'.format(iterate, objective[iterate], delta))

            if (iterate > nsteps or delta < ftol): 
                cond = True
            iterate += 1                    
        # --------------------------------------------------------------------            

        # replace mu and Sigma per their explicit estimator
        self.B = linalg.inv(self._data['X'].T @ self._data['X']) @ self._data['X'].T @ M
        self.Sigma = Sigma

        objective = objective[0:iterate, 0]
        vloglik, gradient = self.variational_objective(M, S, mu, self.Sigma)
        results = {'means' : M, 'variances' : S, 'criterion' : vloglik, 'gradient' : gradient, 'objective': objective}
        return results        

# class rank_constrained(model):
#     
#     def __init__(self, data, rank) :
#         
#         #initializing the data
#         self._data = data
#         self._n    = data['Y'].shape[0]
#         self._q    = rank
#         
#         # initializing the model
#         super(rank_constrained, self).__init__(self._data['Y'].shape[1], self._data['X'].shape[1])
# 
#     def variational_objective(self, M, S, mu, Sigma) :
#     
#         Omega = linalg.inv(Sigma)
#     
#         # A. terms of the objective function
#         # log p(Y|Z) integrated over E_q, with q the variational distribution
#         E_log_p_Y_Z = np.sum(self._data['Y'] * (self._data['O'] + M) - np.exp(self._data['O'] + M + .5 * S)) - self._data['KY']
#      
#         # log p(Z) integrated over E_q, with q the variational distribution
#         # with plugged estimator of Sigma and mu
#         E_log_p_Z = -.5 * (self._p + self._n * np.linalg.slogdet(Sigma)[1] + self._n * self._p * np.log(2 * np.pi))
#         
#         # Entropy of the variational distribution q
#         H_q_Z = .5 * np.sum(np.log(S)) + self._n * self._p * (1 + np.log(2 *np.pi))
#     
#         # B. terms of the gradient
#         grd_M = (M - mu) @ Omega - self._data['Y'] +  np.exp(self._data['O'] + M + .5 * S) 
#         grd_S = .5 * (np.outer(np.ones(self._n), np.diag(Omega)) + np.exp(self._data['O'] + M + .5 * S) - 1/S)    
#       
#         return -(E_log_p_Y_Z + E_log_p_Z + H_q_Z), utils._variational_mat2vec(grd_M, grd_S)
# 
#     def fit_variational_objective(self, M, S, solver = 'ccsa', lb_variance= 1e-4, ftol = 1e-6, xtol = 1e-4, nsteps = int(1e4), lr = 1e-2, use_hessian = 'false'):
# 
#         mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
#         
#         lower_bound = np.concatenate((np.repeat(-np.inf,self._n*self._p), np.repeat(lb_variance, self._n*self._p)))
# 
#         cond = False
#         iterate = 0
#         objective = np.ndarray([nsteps, 1])
#         
#         ## initialize the NLopt optimzer
#         optimizer = nlopt.opt(nlopt.LD_CCSAQ, 2 * self._n * self._p)
#         optimizer.set_lower_bounds(lower_bound)
#         optimizer.set_xtol_rel(xtol)
#         optimizer.set_ftol_abs(0)
#         optimizer.set_ftol_rel(ftol)
# 
#         while cond == False:                                
# 
#             # E step
#             def loss_and_grad(x, grad):
#                 M, S      = utils._variational_vec2mat(x, self._n, self._p)
#                 objective, grad[:] = self.variational_objective(M, S, mu, Sigma)        
#                 return objective
# 
#             optimizer.set_min_objective(loss_and_grad)
#             x0 = np.concatenate((M.flatten(), S.flatten()))
#             x = optimizer.optimize(x0)
# 
#             # M step
#             M, S      = utils._variational_vec2mat(x, self._n, self._p)
#             mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
# 
#             ## assessing convergence            
#             objective[iterate]=  optimizer.last_optimum_value()
#             delta = np.abs(objective[iterate] - objective[iterate-1])/np.abs(objective[iterate])
#             print('Step # {}, loss: {}, delta: {}'.format(iterate, objective[iterate], delta))
# 
#             if (iterate > nsteps or delta < ftol): 
#                 cond = True
#             iterate += 1                    
#         # --------------------------------------------------------------------            
# 
#         # replace mu and Sigma per their explicit estimator
#         self.B = linalg.inv(self._data['X'].T @ self._data['X']) @ self._data['X'].T @ M
#         self.Sigma = Sigma
# 
#         objective = objective[0:iterate, 0]
#         vloglik, gradient = self.variational_objective(M, S, mu, self.Sigma)
#         results = {'means' : M, 'variances' : S, 'criterion' : vloglik, 'gradient' : gradient, 'objective': objective}
#         return results        

    # def loglikelihood() :
    #     """ 
    #         function to compute the log-likelihood of the PLN model
      
    #         Returns: 
    #             res: evaluation of the log-likelihood at this point
    #         """
    #     ## TODO: try direct integration
    #     return None

    # def complete_loglikelihood(self, Z) :
    #     """ 
    #         function to compute the data-complete log-likelihood of the PLN model
    #             Z     (n x p 2D array)  : matrix of observed latent Gaussian vector
  
    #         Returns: 
    #             res: evaluation of the complete data log-likelihood at this point
    #         """
            
    #     log_p_Y_Z = np.sum(self._data['Y'] * (self._data['O'] + Z) - np.exp(self._data['O'] + Z)) - self._data['KY']
    #     # checked to be equivalent to and slightly faster as np.sum(stats.poisson.logpmf(Y, np.exp(Z)))
        
    #     mu = self._X @ self.B
    #     Omega = np.linalg.inv(self.Sigma)
    #     log_p_Z   = -.5 * (np.trace(Omega @ (Z - mu).T @ (Z - mu) ) - self._n * np.linalg.slogdet(Omega)[1]  + self._n * self._p * np.log(2 * np.pi))
    #     # checked to be equivalent to and slightly faster as np.sum(stats.multivariate_normal.logpdf(Z, mu, Sigma))
    
    #     return log_p_Y_Z + log_p_Z

    # def variational_hessian_prod(self, M, S, X_M, X_S, Omega) :
    #     AMS = np.exp(self._data['O'] + M + .5 * S) * (X_M + .5 * X_S)
    #     Hv_M =      AMS + X_M @ Omega
    #     Hv_S = .5* (AMS + X_S /(S**2))
    #     return np.concatenate((Hv_M.flatten(), Hv_S.flatten()))

    # def hessian_prod_variational_scipy(self, x, v):
    #     M, S      = utils._variational_vec2mat(x, self._n, self._p)
    #     S = np.clip(S, 1e-4, np.inf)
    #     X_M, X_S  = utils._variational_vec2mat(v, self._n, self._p)
    #     mu, Sigma = utils._variational_model_params(M, S, self._data['projX'])
    #     Omega = linalg.inv(Sigma)
    #     return self.variational_hessian_prod(M, S, X_M, X_S, Omega)

    # def laplace_loss_and_grad(self, Z, mu, Sigma) :
    #     """ compute the laplace approximation of the log-likelihood of the PLN model """
    #     Omega = linalg.inv(Sigma)   
    #     log_p_Y_Z = np.sum(self._data['Y'] * (self._data['O'] + Z) - np.exp(self._data['O'] + Z)) - self._data['KY']
    #     log_p_Z   = -.5 * (np.trace(Omega @ (Z - mu).T @ (Z - mu) ) - self._n * np.linalg.slogdet(Omega)[1]  + self._n * self._p * np.log(2 * np.pi))

    #     logDetH = 0 
    #     grd_detH = np.zeros((self._n, self._p))
    #     ii = 0
    #     for eZi in np.vsplit(np.exp(Z), Z.shape[0]): 
    #         logDetH += np.linalg.slogdet(Omega + np.diag(eZi))[1]
    #         grd_detH[ii, :] = np.dot(linalg.inv(Omega + np.diag(eZi)), eZi.T).reshape(self._p)
    #         ii += 1
        
    #     grd_Z = (Z - mu) @ Omega - self._data['Y'] +  np.exp(self._data['O'] + Z) + .5 * grd_detH 

    #     return -(log_p_Y_Z + log_p_Z - .5 * logDetH + self._p * .5 * np.log(2 * np.pi)), grd_Z.flatten()

    # def loss_laplace_nlopt(self, x, grad):
    #     Z = x.reshape(self._n, self._p)
    #     mu, Sigma= utils._laplace_model_params(Z, self._data['projX'])
    #     objective, grad[:] = self.laplace_loss_and_grad(Z, mu, Sigma)
        
    #     return objective

    # def loss_laplace_scipy(self, x):
    #     Z = x.reshape(self._n, self._p)
    #     mu, Sigma= utils._laplace_model_params(Z, self._data['projX'])
    #     return self.laplace_loss_and_grad(Z, mu, Sigma)

            
    # def fit_laplace(self, Z0, solver = 'ccsa', lb_variance= 1e-4, ftol = 1e-6, xtol = 1e-4, nsteps = int(1e4), lr = 1e-2, verbose = 0):
    
    #     x0 = Z0.flatten()
                 
    #     if (solver == 'ccsa'):
    #         optimizer = nlopt.opt(nlopt.LD_CCSAQ, x0.size)
    #         optimizer.verbose = 1;
    #         optimizer.set_xtol_rel(xtol)
    #         optimizer.set_ftol_abs(0)
    #         optimizer.set_ftol_rel(ftol)
    #         optimizer.set_min_objective(self.loss_laplace_nlopt)
    #         x = optimizer.optimize(x0)
    #         objective = optimizer.last_optimum_value()
    #         results = optimizer.last_optimize_result()
    #     # --------------------------------------------------------------------            
    #     else : #(solver == 'tnc'):
    #         results = optimize.minimize(
    #             fun     = self.loss_laplace_scipy,
    #              x0      = x0,
    #              jac     = True,
    #              method  = solver, 
    #              options = {'ftol': ftol, 'disp': True })
    #         x = results.x
    #         objective = results.fun        
    #     # --------------------------------------------------------------------            
        
    #     # replace mu and Sigma per their explicit estimator
    #     Z = x.reshape(self._n, self._p)
    #     mu, Sigma= utils._laplace_model_params(Z, self._data['projX'])

    #     self.B = linalg.inv(self._data['X'].T @ self._data['X']) @ self._data['X'].T @ Z
    #     self.Sigma = Sigma
        
    #     lloglik, gradient = self.laplace_loss_and_grad(Z, mu, Sigma)
    #     gradient = gradient.reshape(self._n, self._p)
    #     results = {'latent' : Z, 'criterion' : lloglik, 'gradient' : gradient, 'objective': objective, 'results': results}
    #     return results        
