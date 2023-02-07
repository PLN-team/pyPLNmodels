import pandas as pd
import torch
from pyPLNmodels.VEM import ZIPLN, PLN, PLNPCA, profiled_PLN
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

Y = pd.read_csv("./example_data/test_data/Y_test.csv")
covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
O = (pd.read_csv("./example_data/test_data/O_test.csv"))
true_Sigma = torch.from_numpy(pd.read_csv("./example_data/test_data/true_parameters/true_Sigma_test.csv").values)
true_beta = torch.from_numpy(pd.read_csv("./example_data/test_data/true_parameters/true_beta_test.csv").values)

profiled = profiled_PLN()
profiled.fit(Y,covariates, O)
# plnpca = PLNPCA(q = 5)
# plnpca.fit(Y,O)
# print(plnpca)




pln = PLN()
pln.fit(Y, covariates, O )
# print(pln)

