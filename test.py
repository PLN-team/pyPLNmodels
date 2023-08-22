from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")

add_const = True
# (
#     simu,
#     exog,
#     offsets,
#     true_Sigma,
#     true_beta,
#     true_infla,
# ) = get_simulated_count_data(
#     return_true_param=True, n_samples=200, zero_inflated=True, nb_cov=0, add_const=add_const, return_latent_variables = True
# )
# Y = simu[3]
# ksi = simu[2]
# endog = simu[0]

endog = get_real_count_data()
endog = torch.from_numpy(endog)
offsets = None
exog = None
print("percentage zeros:", torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1]))
# full = ZIPln(endog=endog, exog=exog, offsets=offsets)
full = ZIPln(
    endog,
    offsets=offsets,
    exog=exog,
    add_const=add_const,
    use_closed_form_prob=True,  # Y_inflated = true_Y, true_covariance= true_Sigma, true_coef=true_beta, true_infla=true_infla, ksi = ksi
)
full.fit(nb_max_iteration=1550, tol=0)
full.show()
# print(full)
# full.show()
# beta_zi = full._coef.detach()
print("mse beta:", torch.mean((full._coef - true_beta) ** 2))
print("mse infla:", torch.mean((full._coef_inflation - true_infla) ** 2))
print("mse Sigma:", torch.mean((full._covariance - true_Sigma) ** 2))


def show_mses(model):
    absc = np.arange(len(model.mse_cov_list))
    plt.plot(absc, model.mse_cov_list, label="Sigma")
    plt.plot(absc, model.mse_beta_list, label="beta")
    plt.plot(absc, model.mse_infla_list, label="infla")
    plt.plot(absc, model.mse_ksi_list, label="ksi")
    plt.legend()
    plt.yscale("log")
    plt.show()


fig, axes = plt.subplots(2)
sns.heatmap(full._coef.detach(), ax=axes[0])
sns.heatmap(true_beta, ax=axes[1])
plt.show()

fig, axes = plt.subplots(2)
sns.heatmap(full._coef_inflation.detach(), ax=axes[0])
sns.heatmap(true_infla, ax=axes[1])
plt.show()
show_mses(full)


# sns.heatmap(true_Sigma)
# plt.show()


# pln = Pln(endog=endog, offsets = offsets, exog = exog, add_const = add_const)
# pln.fit()
# pln.show()
# sns.heatmap(pln.covariance - full.covariance)
# print("mse :", torch.mean((pln._coef - true_beta) ** 2))
# plt.show()
# sns.heatmap(full._latent_prob.detach())
# plt.show()
# sns.heatmap(simu[2])
# plt.show()
# sns.heatmap(full._latent_prob.detach() - simu[2])
# plt.show()
print("mean", torch.mean((full._latent_prob - simu[2]) ** 2))
print(
    "mean random",
    torch.mean(
        (simu[2] - torch.randint(size=(exog.shape[0], true_beta.shape[1]), high=1)) ** 2
    ),
)
