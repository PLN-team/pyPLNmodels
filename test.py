from pyPLNmodels.VEM import PLNPCA, PLN, _PLNPCA, _PLNPCA_noS
from pyPLNmodels._utils import get_simulated_count_data, get_real_count_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir("pyPLNmodels/")


n = 100
p = 50
q = 3
d = 1

# counts,covariates,offsets, true_Sigma, true_beta= get_simulated_count_data(return_true_param=True)
counts = get_real_count_data()


pca = _PLNPCA(q)

pca.fit(Y)
print(pca)
