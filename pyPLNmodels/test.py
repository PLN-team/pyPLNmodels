import pandas as pd 
import torch 
from VEM import ZIPLN, PLNnoPCA, PLNPCA
import matplotlib.pyplot as plt 
import numpy as np


Y = pd.read_csv("../example_data/test_data/Y_test.csv")
O = np.exp(pd.read_csv("../example_data/test_data/O_test.csv"))
covariates = pd.read_csv("../example_data/test_data/cov_test.csv")
true_Sigma = torch.from_numpy(pd.read_csv("../example_data/test_data/true_parameters/true_Sigma_test.csv").values)
true_beta = torch.from_numpy(pd.read_csv("../example_data/test_data/true_parameters/true_beta_test.csv").values)



pln = PLNnoPCA()
pln.fit(Y,O,covariates)
print(pln)

plnpca = PLNPCA(q = 5)
plnpca.fit(Y,O,covariates)
print(plnpca)
