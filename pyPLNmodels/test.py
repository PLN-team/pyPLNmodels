import pandas as pd 
import torch 
from VEM import ZIPLN, PLNnoPCA, PLNPCA
import matplotlib.pyplot as plt 
import numpy as np


Y = pd.read_csv("../example_data/Y_test")
O = np.exp(pd.read_csv("../example_data/O_test"))
covariates = pd.read_csv("../example_data/cov_test")
true_Sigma = torch.from_numpy(pd.read_csv("../example_data/true_Sigma_test").values)
true_beta = torch.from_numpy(pd.read_csv("../example_data/true_beta_test").values)



pln = PLNnoPCA()
pln.fit(Y,O,covariates)
print(pln)

plnpca = PLNPCA(q = 5)
plnpca.fit(Y,O,covariates)
print(plnpca)
