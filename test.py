from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")

(
    endog,
    exog,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(return_true_param=True, n_samples=200, zero_inflated=True)


print("percentage zeros:", torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1]))
# sns.heatmap(true_Sigma)
# plt.show()
# pln = ZIPLN(true_infla=true_infla, true_covariance=true_Sigma, true_coef=true_beta)
# pln.fit(counts, covariates, offsets, nb_max_iteration=250)
# pln.print_mse()
# sns.heatmap(pln._covariance.detach())
# plt.show()
# sns.heatmap(true_Sigma)
# plt.show()
full = ZIPln(endog=endog, exog=exog, offsets=offsets)
full.fit(nb_max_iteration=200)


def MSE(t):
    return torch.mean(t**2)


print("mse:", MSE(true_beta - full._coef))
# print(full)
# full.show()
