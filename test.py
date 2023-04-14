from pyPLNmodels.VEM import PLNPCA, PLN, _PLNPCA, _PLNPCA_noS
from pyPLNmodels._utils import get_simulated_count_data, get_real_count_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir("pyPLNmodels/")

Y, covariates, O, true_Sigma, true_beta = get_simulated_count_data(
    return_true_param=True, p=100, rank=25
)

q = 10
lr = 0.01
# nb_max = 1000
pca = _PLNPCA(q, true_Sigma, true_beta)
tol = 0.0000001
pca.fit(Y, covariates, O, tol=tol, lr=lr)
# pca.show()

# pca = _PLNPCA(q)
# pca.fit(Y, tol = 0.000000001)
# pca.show()
nospca = _PLNPCA_noS(q, true_Sigma, true_beta)
nospca.fit(Y, covariates, O, tol=tol, lr=lr)
# nospca.show()
# abscisse_pca = np.arange(len(pca.elbos_list))
abscisse_pca = pca.plotargs.running_times
abscisse_nospca = np.arange(len(nospca.elbos_list))
abscisse_nospca = nospca.plotargs.running_times
fig, axes = plt.subplots(2)

axes[0].plot(abscisse_pca, pca.elbos_list, label="pca")
axes[0].plot(abscisse_nospca, nospca.elbos_list, label="nospca")
axes[0].legend()


axes[1].plot(
    abscisse_pca, pca.mse_Sigma_list, label="pca Sigma", color="red", linestyle="--"
)
axes[1].plot(
    abscisse_pca, pca.mse_beta_list, label="pca beta", color="blue", linestyle="--"
)
axes[1].plot(abscisse_nospca, nospca.mse_Sigma_list, label="noS Sigma", color="red")
axes[1].plot(abscisse_nospca, nospca.mse_beta_list, label="noS beta", color="blue")
axes[1].legend()
plt.show()
