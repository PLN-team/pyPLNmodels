from VEM import PLNnoPCA, PLNPCA
import pandas as pd 
import torch
import numpy as np 
from IMPSPLNmodel import IMPSPLN
import matplotlib.pyplot as plt 

Y = pd.read_csv('../example_data/Y_test')
O = np.exp(pd.read_csv('../example_data/O_test'))
covariates = pd.read_csv('../example_data/cov_test')
true_Sigma = torch.from_numpy(pd.read_csv('../example_data/true_Sigma_test').values)
true_beta = torch.from_numpy(pd.read_csv('../example_data/true_beta_test').values)
n = 200
#pln = PLNnoPCA()
#pln.fit(Y,O,covariates) 
#print(pln)
#plt.show()
#pca = PLNPCA(q = 5)
#pca.fit(Y,O,covariates, lr = 0.1)
#print(pca)
#plt.show()
imps = IMPSPLN(q = 5)
imps.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= 10,criterionMax= 300, batchSize=n, nbMonteCarloSamples= 500)
print('mean mean', torch.mean(torch.tensor(imps.listMeanESS)))
plt.plot(np.arange(0, len(imps.listMeanESS)), imps.listMeanESS)
plt.yscale('log')
plt.show()  
#print(imps)
#plt.show()



