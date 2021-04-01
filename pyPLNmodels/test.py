import pandas as pd 
import torch 
from IMPSPLNmodel import IMPSPLN
import matplotlib.pyplot as plt 
from utils import lissage 
import numpy as np


Y = pd.read_csv("../example_data/Y_test")
O = np.exp(pd.read_csv("../example_data/O_test"))
covariates = pd.read_csv("../example_data/cov_test")
true_Sigma = torch.from_numpy(pd.read_csv("../example_data/true_Sigma_test").values)
true_beta = torch.from_numpy(pd.read_csv("../example_data/true_beta_test").values)
n = 20

nbEpochMax = 30
lr = 0.1 
q = 10


imps = IMPSPLN(q=q)
imps.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=30,
    batchSize=n,
    nbMonteCarloSamples=100,
    method="recycling",
    lr=lr,
)
print(imps)

