import numpy as np
import torch
import scipy.linalg as SLA 


class Poisson_reg():
    '''
    Poisson regressor class. The purpose of this class is to initialize the PLN model.
    '''
    
    def __init__(self): 
        pass
    

    
    def fit(self,O,X,Y, Niter_max = 300, tol = 0.1, lr = 0.00008,  verbose = False): 
        '''
        We run a gradient ascent to maximize the log likelihood. We do this by hand : we compute the gradient ourselves. 
        The log likelihood considered is the one from a poisson regression model. It is the same as PLN without the latent layer Z. 
        We are only trying to have a good guess of beta before doing anything. 
        
        args : 
                '0' : offset, size (n,p)
                'X' : covariates, size (n,p)
                'Y' : samples , size (n,p)
                'Niter_max' :int  the number of iteration we are ready to do 
                'tol' : float. the tolerance criteria. We will stop if the norm of the gradient is less than 
                       or equal to this threshold
                'lr' : float. learning rate for the gradient ascent
                'verbose' : bool. if True, will print some stats on the 
                
        returns : None but update the parameter beta 
        '''
        
        #we initiate beta 
        beta = torch.rand(X.shape[1])
        i = 0
        grad_norm = 2*tol
        while i<Niter_max and  grad_norm > tol : # condition to keep going
            grad = grad_poiss_beta(O,X,Y,beta) # computes the gradient 
            grad_norm = torch.norm(grad) 
            beta += lr*grad_poiss_beta(O,X,Y,beta)# update beta 
            i+=1
            
            # some stats if we want some 
            if verbose == True : 
                if i % 10 == 0 : 
                    print('log likelihood  : ', compute_l(0,X,Y,beta))
                    print('Gradient norm : ', grad_norm)
        if i < Niter_max : 
            print('---------------------Tolerance reachedin {} iterations'.format(i))
        else : 
            print('---------------------Maximum number of iterations reached')
        print('----------------------Gradient norm : ', grad_norm)
        self.beta = beta # save beta 
        
        
    def fit_torch(self,O,X,Y, Niter_max = 300, tol = 0.1, lr = 0.0001, verbose = False): 
        '''
        Does exaclty the same as fit() but uses autodifferentiation of pytorch. 
        '''
        
        beta = torch.rand(X.shape[1], requires_grad = True)
        optimizer = torch.optim.Adam([beta], lr = lr)
        i = 0
        grad_norm = 2*tol
        while i<Niter_max and  grad_norm > tol :
            loss = -compute_l(O,X,Y,beta)
            loss.backward()
            optimizer.step()
            grad_norm = torch.norm(beta.grad)
            beta.grad.zero_()
            i+=1
            if verbose == True : 
                if i % 10 == 0 : 
                    print('log like : ', -loss)
                    print('grad_norm : ', grad_norm)
        if verbose :
            if i < Niter_max : 
                print('-------------------Tolerance reached in {} iterations'.format(i))
            else : 
                print('-------------------Maxium number of iterations reached')
        self.beta = beta 
    

def grad_poiss_beta(O,X,Y,beta): 
    return torch.sum(-torch.multiply(X,torch.exp(O+X@beta).reshape(-1,1))+torch.multiply(Y.reshape(-1,1),X),dim = 0)
    
def compute_l(O,X,Y,beta):
    return torch.sum(-torch.exp(O + X@beta)+torch.multiply(Y,O+X@beta))    
    
def sample(O,X,true_beta):
        parameter = np.exp(O + X@true_beta)
        return torch.poisson(parameter)