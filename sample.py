import numpy as np
import torch
import scipy.linalg as SLA 



class sample_PLN(): 
    '''
    simple class to sample some variables with the PLN model. 
    The main method is the sample one, however we can also plot the data calling the plot_Y method. 
    The method conditional prior should not be used and have not been tested properly. 
    '''
    
    def __init__(self): 
        pass 

    def sample(self, Sigma, beta, O, covariates): 
        '''
        sample Poisson log Normal variables. 
        The number of samples is the the first size of O, the number of species
        considered is the second size of O
        The number of covariates considered is the first size of beta. 
        
        '''
        self.Sigma = Sigma # unknown parameter in practice
        self.beta = beta #unknown parameter in practice
        
        self.O = O 
        self.covariates = covariates
        
        self.Z = torch.stack([self.Sigma@np.random.randn(p) for _ in range(n)])
        
        self.n = self.O.shape[0]
        self.p = self.Sigma.shape[0]
        self.d = self.covariates.shape[1]
        
        parameter = torch.exp(self.O + self.covariates@self.beta + self.Z)
        self.Y = np.random.poisson(lam = parameter)
        return self.Y 
        #return parameter.numpy()
    def plot_Y(self): 
        '''
        plot all the Y_ij sampled before. There will be n*p values in total. The color represent the site number. 
        Note that we need to have called self.sample() before otherwise it won't print anything 
        '''
        color = np.array([[site]*self.p for site in range(self.n) ]).ravel()*10
        plt.scatter(np.arange(0,self.n*self.p),self.Y.ravel(), c = color, label = 'color = site number')
        plt.legend()
        plt.ylabel('count number')
        plt.show()

    def conditionalprior(self): 
        mu = self.O[0,0]
        functions = list()
        for i in range(self.n): 
            mu_i = self.covariates[i].dot(self.beta[0])
            functions.append(lambda z : -z**2/(2*self.Sigma[0,0]**2)-np.exp(mu_i+z)+float(self.Y[i])*(mu_i+z))
        return functions 