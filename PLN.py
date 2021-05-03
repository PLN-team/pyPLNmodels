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
import time

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

    
class PLNmodel(): 
    '''
    PLN model. The goal of this class is to compute the parameter beta and Sigma of the PLN model. 
    We use here variationnal approximation since we can't compute the log likelihood of the 
    latent variables given the data. 
    '''
    
    
    def __init__(self, C_init, beta_init, M_init, S_init): 
        
        '''
            Initialization : 
            'Y' : the data, size (n,p). n is the number of samples we have and p the number of species. 
                  THE TYPE IS INT
            'O': offset : additional offset. (not very important for comprehension). size (n,p)
            'covariates' : covariates, size (n,d)
            'Sigma_init' : initialization for Sigma. I plan to do a more advanced initialization. 
            'beta_init ' : Initialization for beta. I plan to do a more advanced initialization. 
            'M_init' : initialization for the variational parameter M
            'S_init ': initialization for the variational parameter S
        '''
        
        # model parameters
        self.Sigma = torch.clone(torch.mm(C_init,C_init.T))
        #self.Sigma.requires_grad_(True)
        self.beta = torch.clone(beta_init)
        self.beta.requires_grad_(True)
        
        #variational parameters
        self.M = torch.clone(M_init)
        self.M.requires_grad_(True)
        self.S = torch.clone(S_init) 
        self.S.requires_grad_(True)
        
        # some useful variables
        self.det_Sigma = torch.det(self.Sigma)
        self.inv_Sigma = torch.inverse(self.Sigma)
        
        # optimizer for the VE_step
        self.VE_step_optimizer = torch.optim.Adam([self.S,self.M], lr = 0.002)
        self.VE_step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.VE_step_optimizer, patience = 3, factor = 0.9)
        
        #optimizer for the M_step
        self.M_step_optimizer = torch.optim.Adam([self.beta], lr = 0.01)
        self.M_step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.M_step_optimizer, patience = 3, factor = 0.9)        
        
        #optimizer for a full gradient ascent
        self.full_optimizer = torch.optim.Adam([self.S,self.M,self.beta])
        
        # just to stocke the parameters
        self.params = {'M': self.M, 'S' : self.S, 'beta' : self.beta}
        
        # to stock the 
        self.MSE_Sigma_list = list()
        self.MSE_beta_list = list()
        self.ELBO_list = list()
        
        self.time = list()

        
    # we define here the gradients computed manually as a sanity check. We can check that they are equals 
    # to the gradients computed with the autodifferentiation of pytorch (diff = 1e-15)
    def grad_Sigma(self): 
        with torch.no_grad():
            self.inv_Sigma = torch.inverse(self.Sigma)
            grad = -self.n/2*(self.inv_Sigma)# + torch.diag(torch.diagonal(self.inv_Sigma))) on a enlevé car avec ça ca match avec pytorch. 
            grad += 1/2*(sum([self.inv_Sigma@(torch.outer(self.M[i,:],self.M[i,:])+ torch.diag(self.S[i,:]))@self.inv_Sigma  for i in range(self.n)]))
        return grad
    def grad_M(self): 
        with torch.no_grad():
            grad = -torch.mm(self.M,self.inv_Sigma)
            grad -= torch.exp(self.O + torch.mm(self.covariates,self.beta) + self.M + torch.pow(self.S,2)/2)
            grad += self.Y 
        return grad 
    def grad_S(self):
        with torch.no_grad():
            grad = -1/2*torch.mm(torch.ones((self.n,self.p)), torch.diag(torch.diag(self.inv_Sigma)))
            grad-= torch.mul(self.S,torch.exp(self.O + torch.mm(self.covariates,self.beta) + self.M + torch.pow(self.S,2)/2))
            grad += 1/2*torch.div(1,self.S)
        return grad 
    def grad_beta(self): 
        with torch.no_grad():
            grad = - torch.mm(self.covariates.T,torch.exp( self.O + self.M + torch.pow(self.S,2)/2 + torch.mm(self.covariates,self.beta)))
            grad += torch.mm(self.covariates.T,self.Y.double())
        return grad 
    
        
    def extract_data(self,data): 
        '''
        function to extract the data. This function is just here to have a code more compact. 
        
        args : 
              'data': list with 3 elements : Y, O and covariates in this order. 
        '''
        #known variables
        self.Y = data[0];self.O = data[1];self.covariates = data[2]
        self.n, self.p = self.Y.shape
    
    def train_step(self, optimizer, Y_,covariates_, O_,M_,S_,Sigma_, beta_): 
        '''
        do one step on gradient ascent. We will optimize only the parametrs of the optimizer. 
        
        args : 
                'optimizer' torch.optim.optimizer object. this function will update the parameters of this optimizer. 
                
                'Y_' size (batch_size, p). 
                'covariates' size (batch_size, d)
                'O_' size (batch_size, p)
                'M_' size (batch_size, p)
                'S_' size (batch_size, p)
                'Sigma_' size (p,p)
                'beta_' size(batch_size, p)
                
        returns : 
        
                'loss' float. the loss computed. correspond to -ELBO 
        '''
        optimizer.zero_grad()
        loss = -self.compute_ELBO(Y_, covariates_, O_, M_, S_,Sigma_,beta_)
        loss.backward()
        if torch.isnan(loss).item() == True : 
            print('NAN')
        else : self.last_param = self.params 
        optimizer.step()
        return loss 

   
    def get_batch(self,batch_size): 
        '''
        get the batches required to do a  minibatch gradient ascent. the batches are generate for the
        variables Y covariates 0 M and S (only those who depends on n). It is a generator to handle memory. 
        
        args : 
                'batch_size' int.  the batch size you want. 
                
        returns : a generator. Will generate n/batch_size samples of size batch_size (except the last one since the rest of the division is
                    not always an integer)
        '''
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        nb_full_batch, last_batch_size  = self.n//batch_size, self.n % batch_size  
        self.batch_size = batch_size
        for i in range(nb_full_batch): 
            yield   (self.Y[indices[i*batch_size: (i+1)*batch_size]], 
                    self.covariates[indices[i*batch_size: (i+1)*batch_size]],
                    self.O[indices[i*batch_size: (i+1)*batch_size]], 
                    self.M[indices[i*batch_size: (i+1)*batch_size]], 
                    self.S[indices[i*batch_size: (i+1)*batch_size]])
                  
        if last_batch_size != 0 : 
            self.batch_size = last_batch_size
            yield   (self.Y[indices[-last_batch_size:]], 
                    self.covariates[indices[-last_batch_size:]],
                    self.O[indices[-last_batch_size:]],
                    self.M[indices[-last_batch_size:]], 
                    self.S[indices[-last_batch_size:]])
            
    

    def print_stats(self, loss, params, optimizer): 
        '''
        small function that print some stats. 
        
        It will print the actual learning rate of the optimizer, the actual log likelihood 
        and the norms of each parameter's gradient. The norm of the parameter's gradient should be low
        when we are close to the optimum. 
        '''
        print('---------------------------------lr :', optimizer.param_groups[0]['lr'])
        print('---------------------------------log likelihood :', - loss.item())
        for param_name, param in params.items(): 
            print('---------------------------------grad_{}_norm : '.format(param_name), round(torch.norm(param.grad).item(), 3))
    
    
    def VEM(self,data, number_VEM_step ,batch_size,   tolerance = 0.1, beginning_VE_step_lr = 0.002, beginning_M_step_lr = 0.01, 
                requires_init = False,N_epoch_VE = 50, N_epoch_M = 75, verbose = False): 
        '''
        function to optimize both the variational parameters and the model parameters.
        We alternate between two steps : Variational step (VE_step) and Maximization step (M_step). 
        
        
        args : 
            'number_VEM_step' : int . Number of times we want to do the VEM step, i.e. alternate between VE step and M step. 
                                The greater the better the approximation, the greater the longer time it takes. 
            
            'beginning_VE_step_lr' : float. the beginning of the learning for the VE_step. The VE will start with this lr. 
            'beginning_M_step_lr' : float. Same for beta, the M step will start with this lr. 
            
       returns : 
               M_S_lr, beta_lr : the learning rates of both steps, so that we can continue after that with the appropriate learning rates.  
        ''' 
        
        self.running_time = time.time()
        
        # we first extract the data. 
        self.extract_data(data)
        
        self.ELBO_list.append(1)
        
        if requires_init : 
            print('Initialisation ... ')
            clf = Poisson_reg()
            for j in range(p): 
                Y_j = self.Y[:,j]   
                O_j = self.O[:,j]
                
                clf.fit_torch(O_j,self.covariates,Y_j, verbose = False , Niter_max = 500, lr = 0.1)
                with torch.no_grad():
                    self.beta[:,j] = clf.beta
            print('Initialisation finished')
            
        # we do as many VEM_step we are asked to. 
        for i in range(number_VEM_step): 
            #M STEP-------UPDATE------- None N_epoch :VE  100
            # closed form for Sigma, we don't need to optimize
            with torch.no_grad(): 
                #new
                #self.Sigma = 1/self.n*(torch.sum(torch.stack([torch.outer(self.M[i,:],self.M[i,:]) + torch.diag(torch.multiply(self.S,self.S)[i,:])  for i in range(self.n)]), axis = 0))
                #before
                self.Sigma = 1/self.n*(torch.sum(torch.stack([torch.outer(self.M[i,:],self.M[i,:]) + torch.diag(self.S[i,:])  for i in range(self.n)]), axis = 0))
                #self.beta = torch.mm(torch.inverse(torch.mm(self.covariates.T,self.covariates)),torch.mm(self.covariates.T,self.M))
            #gradient ascent for beta 
            self.torch_gradient_ascent(self.M_step_optimizer, self.M_step_scheduler, 
                                       lr = beginning_M_step_lr, tolerance = tolerance, N_epoch= N_epoch_M, verbose= verbose, batch_size = batch_size )
            #VE STEP
            #gradient ascent for M and S 
            self.torch_gradient_ascent(self.VE_step_optimizer, self.VE_step_scheduler, lr = beginning_VE_step_lr,
                                    tolerance= tolerance, N_epoch= N_epoch_VE, verbose= verbose)
            self.ELBO_list.append(self.current_ELBO)
            if i %  100 == 0 : 
                print('-------UPDATE-------')
                print('ELBO : ', np.round(self.current_ELBO,5))
        
        #keep track of the runningtime 
        self.running_time = time.time()- self.running_time
        return self.VE_step_optimizer.param_groups[0]['lr'],self.M_step_optimizer.param_groups[0]['lr']        
    
    def torch_gradient_ascent(self,optimizer, scheduler, lr = None, tolerance = 2, N_epoch = 500, verbose = True, batch_size = None ): 
        '''
        gradient ascent function. We compute the gradients thanks to the autodifferentiation of pytorch. 
        
        args : 
                'optimizer' : torch.optim.optimizer. the optimizer for the parameters. 
                'scheduler' : torch.optim.lr_scheduler.  scheduler for the optimizer above. 
                
                'lr' : float.  a learning rateM if we want to set the optimizer learning rate to a certain lr. If None, 
                      it will take the actual learning_rate of the optimizer. 
                'tolerance': float. the threshold we set to stop the algorithm. It will stop if the norm of each gradient's parameter 
                             is lower than this threshold, or if we are not improving the loss more than tolerance. 
                'N_epoch': int. the Maximum number of epoch we are ready to do. 
                
                'Verbose' : bool. if True, will print some messages useful to interpret the gradient ascent. If False, nothing will be printed. 
                
                'batch_size' : int or None. If None, the batch size will be n, so it will be a classical vanilla algorithm. 
                              if int, we will split the data set in batch size and do a gradient step for each mini_batch. 
                              
        
        returns : the parameters optimized. 
        '''
        
        
        # we set the gradient to zero just to make sure the gradients are properly calculated
        optimizer.zero_grad()
        
        if lr is not None : # if we want to set a threshold, we set it. Ohterwise, we skip this condition and keep the actual learning_rate
            optimizer.param_groups[0]['lr'] = lr 
            
        #if batch_size is None, we take n. 
        if batch_size == None : 
            batch_size = self.Y.shape[0]
        
        stop_condition = False 
        i = 0
        old_epoch_loss = 1.
        
        while i < N_epoch and stop_condition == False: 
            epoch_loss = 0.
            # we run through the whole dataset. if batch_size was None, this loop contains only one element. 
            t0 = time.time()
            for Y_b, covariates_b, O_b, M_b, S_b in self.get_batch(batch_size): 
                epoch_loss += self.train_step(optimizer, Y_b, covariates_b, O_b, M_b, S_b, self.Sigma,self.beta)
            self.time.append(time.time()-t0)
            if verbose and i % 25 == 0 : 
                self.print_stats(epoch_loss, self.params, optimizer)
            i += 1
            scheduler.step(epoch_loss)
            # condition to see if we have reach the tolerance threshold
            if  abs(epoch_loss.item() - old_epoch_loss) < tolerance : #and i > 10 and max([torch.norm(param.grad) for param in params]) < tolerance  or
                #if max([torch.norm(param.grad) for param in params]) < tolerance  or abs(loss.item()- old_loss)>  tolerance :
                stop_condition = True 
            old_epoch_loss = epoch_loss
            
        #keep track of the ELBO 
        self.current_ELBO = -epoch_loss.item()
        
        if verbose : # just print some stats if we want to 
            if stop_condition : 
                print('---------------------------------Tolerance {} reached in {} iterations'.format(tolerance, i))
            else : 
                print('---------------------------------Maximum number of iterations reached : ', N_epoch)
            self.print_stats(epoch_loss, self.params, optimizer)
        return self.last_param

    
    
    
    def compute_ELBO(self, Y, covariates,O,M,S,Sigma,beta): 
        ''' 
        computes the ELBO. We simply apply the formula given above. 
        '''

        batch_size,p = Y.shape
        
        # we store some matrices to avoid computing it two times
        inv_Sigma = torch.inverse(Sigma)
        Gram_matrix = torch.mm(covariates,beta) 
        help_calculus = O + Gram_matrix + M 
        tmp = -batch_size/2*torch.log(torch.det(Sigma)) #-1/2*( torch.sum(torch.mm(torch.mm(M,inv_Sigma),M.T).diagonal()))
        
        #term = -torch.exp(help_calculus+ torch.pow(S,2)/2)
        #mask = torch.isnan(term)
        #print('nb elements nonzero', torch.sum(mask))
        tmp += torch.sum(-torch.exp(help_calculus+ torch.pow(S,2)/2) + torch.multiply(Y, help_calculus))
        tmp -= 1/2*torch.trace(torch.mm(torch.mm(M.T, M) + torch.diag(torch.sum(S, dim = 0)), inv_Sigma))
        #tmp-= batch_size*p/2
        tmp += 1/2*torch.sum(torch.log(torch.prod(S, dim = 1)))
        return tmp
    def compute_ELBO_bis(self, Y, covariates,O,M,S,Sigma,beta): 
        ''' 
        computes the ELBO. We simply apply the formula given above. 
        '''
        batch_size,p = Y.shape
        if torch.min(S)<0 : 
            print('neg')
        # we store some matrices to avoid computing it two times
        inv_Sigma = torch.inverse(Sigma)
        Gram_matrix = torch.mm(covariates,beta) 
        help_calculus = O + Gram_matrix + M 
        tmp = -batch_size/2*torch.log(torch.det(Sigma)) #-1/2*( torch.sum(torch.mm(torch.mm(M,inv_Sigma),M.T).diagonal()))
        
        #term = -torch.exp(help_calculus+ torch.pow(S,2)/2)
        #mask = torch.isnan(term)
        #print('nb elements nonzero', torch.sum(mask))
        tmp += torch.sum(-torch.exp(help_calculus+ torch.pow(S,2)/2) + torch.multiply(Y, help_calculus))
        tmp -= 1/2*torch.trace(torch.mm(torch.mm(M.T, M) + torch.diag(torch.sum(torch.multiply(S,S), dim = 0)), inv_Sigma))
        #tmp-= batch_size*p/2
        tmp += 1/2*torch.sum(torch.log(torch.prod(torch.multiply(S,S), dim = 1)))
        return tmp

    def full_grad_ascent(self, data, lr = None, tolerance = 2, N_iter = 500, verbose = True): 
        '''
        gradient ascent function. We compute the gradients thanks to the autodifferentiation of pytorch. 
        
        args :  'data' the data i.e. a list of 3 elements Y O and covariates in this order
                
                'lr' : float.  a learning rate if we want to set the optimizer learning rate to a certain lr. If None, 
                      it will take the actual learning_rate of the optimizer. 
                'tolerance': float. the threshold we set to stop the algorithm. It will stop if the norm of each gradient's parameter 
                             is lower than this threshold, or if we are not improving the loss more than tolerance. 
                'N_iter': int. the Maximum number of iterations we are ready to do. 
                
                'Verbose' : bool. if True, will print some messages useful to interpret the gradient ascent. If False, nothing will be printed. 

        
        returns : the parameters optimized. 
        '''
        self.extract_data(data)
        
        # we set the gradient to zero just to make sure the gradients are properly calculated
        self.full_optimizer.zero_grad()
        
        if lr is not None : # if we want to set a threshold, we set it. Ohterwise, we skip this condition and keep the actual learning_rate
            self.full_optimizer.param_groups[0]['lr'] = lr 
            
        
        stop_condition = False 
        i = 0
        old_loss = 0 
        
        while i < N_iter and stop_condition == False: 
            
            loss = self.train_step(self.full_optimizer, self.Y, self.covariates, self.O, self.M, self.S, self.Sigma, self.beta)
            i += 1 
            
            # condition to see if we have reach the tolerance threshold
            if max([torch.norm(param.grad) for param in self.params.values()]) < tolerance  or abs(loss - old_loss) < tolerance : #and i > 10:
                stop_condition = True 
            old_loss = loss 
            
            #update Sigma with the closed form. 
            with torch.no_grad(): 
                #before
                self.Sigma = 1/self.n*(torch.sum(torch.stack([torch.outer(self.M[i,:],self.M[i,:]) + torch.diag(self.S[i,:])  for i in range(self.n)]), axis = 0))
                #new 
                #self.Sigma = 1/self.n*(torch.sum(torch.stack([torch.outer(self.M[i,:],self.M[i,:]) + torch.diag(torch.multiply(self.S,self.S)[i,:])  for i in range(self.n)]), axis = 0))
            self.MSE_Sigma_list.append(torch.mean((self.Sigma-true_Sigma)**2).item())
            self.MSE_beta_list.append(torch.mean((self.beta-true_beta)**2).item())
            self.ELBO_list.append(-loss)
            
            if verbose and i % 100 == 0 : 
                print('iteration number: ', i)
                self.print_stats(loss, self.params, self.full_optimizer)
                print('-------UPDATE-------')
                print(' MSE with Sigma : ', np.round(self.MSE_Sigma_list[-1],5))
                print(' MSE with beta : ', np.round(self.MSE_beta_list[-1],5))
                print('ELBO : ', np.round(-loss.item(),5))
            


        if verbose : # just print some stats if we want to 
            if stop_condition : 
                print('---------------------------------Tolerance reached in {} iterations'.format(i))
            else : 
                print('---------------------------------Maximum number of iterations reached')
            self.print_stats(loss,self.params, self.full_optimizer)   
        
        return self.params
