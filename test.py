import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyPLNmodels._utils import sample_PLN
from pyPLNmodels import PLNPCA, PLN
import pandas as pd

n = 1000
p = 300
q = 10
d = 1

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


true_beta = torch.randn(d, p, device=DEVICE)
C = torch.randn(p, q, device=DEVICE)/5
O = torch.ones((n, p), device=DEVICE)/2
covariates = torch.ones((n, d), device=DEVICE)*0 + 1
true_Sigma = torch.matmul(C,C.T)
Y, _, _ = sample_PLN(C, true_beta, covariates, O)
# Y = pd.read_csv("./example_data/test_data/Y_test.csv")
# covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
# O = pd.read_csv("./example_data/test_data/O_test.csv")
# true_Sigma = torch.from_numpy(
#     pd.read_csv("./example_data/test_data/true_parameters/true_Sigma_test.csv").values
# )
# true_beta = torch.from_numpy(
#     pd.read_csv("./example_data/test_data/true_parameters/true_beta_test.csv").values
# )
# Y = pd.read_csv("./example_data/real_data/oaks_counts.csv")
# # covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
# covariates = None
# O = np.log(pd.read_csv("./example_data/real_data/oaks_offsets.csv"))

pca = PLNPCA(ranks=[2, 30, 50, 80, 130, 170, 200, 230, 270, 300])
pca.fit(Y, covariates, O, tol=0.0001, lr=0.001, do_smart_init = False)
pca.show()
pln = PLN()
pln.fit(Y, covariates, O)
print("loglike pln", pln.loglike)
