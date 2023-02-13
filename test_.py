import pandas as pd
import torch
from pyPLNmodels.VEM import ZIPLN, PLN, PLNPCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Y = pd.read_csv("./example_data/test_data/Y_test.csv")
covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
O = pd.read_csv("./example_data/test_data/O_test.csv")
true_Sigma = torch.from_numpy(
    pd.read_csv("./example_data/test_data/true_parameters/true_Sigma_test.csv").values
)
true_beta = torch.from_numpy(
    pd.read_csv("./example_data/test_data/true_parameters/true_beta_test.csv").values
)
def MSE(t): 
    return torch.mean(t**2)

def test_find_right_pln_parameters(): 
    pln = PLN()
    pln.fit(Y, covariates, O, nb_max_iteration=20)
    mse =  MSE(pln.beta - true_beta)
## tester avec que des Y, et plus O et plus cov etc. 

