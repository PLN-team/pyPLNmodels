from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")

add_const = True
(
    simu,
    exog,
    offsets,
    true_Sigma,
    true_beta,
    true_infla,
) = get_simulated_count_data(
    return_true_param=True,
    n_samples=20,
    zero_inflated=True,
    nb_cov=0,
    add_const=add_const,
    return_latent_variables=True,
    rank=5,
    dim=5,
)
Y = simu[3]
ksi = simu[2]
endog = simu[0]

# endog = get_real_count_data()
# endog = torch.from_numpy(endog)
# offsets = None
# exog = None
print(
    "percentage zeros inflated:",
    torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1]),
)
print("percentage zeros Y:", torch.sum(Y == 0) / (Y.shape[0] * Y.shape[1]))
# full = ZIPln(endog=endog, exog=exog, offsets=offsets)
nb_iter = 18950
use_closed_form_prob = False
perfect = ZIPln(
    Y,
    offsets=offsets,
    exog=exog,
    Y_inflated=endog,
    add_const=add_const,
    use_closed_form_prob=use_closed_form_prob,
    true_covariance=true_Sigma,
    true_coef=true_beta,
    true_infla=true_infla,
    ksi=ksi,
    perfect_init=False,
)
perfect.fit(nb_max_iteration=nb_iter, tol=0)
# perfect.show()
normal = ZIPln(
    Y,
    offsets=offsets,
    exog=exog,
    add_const=add_const,
    Y_inflated=endog,
    use_closed_form_prob=use_closed_form_prob,
    true_covariance=true_Sigma,
    true_coef=true_beta,
    true_infla=true_infla,
    ksi=ksi,
    perfect_init=False,
)


# normal.fit(nb_max_iteration=10, tol=0)
# normal.show()
# print(full)
# full.show()
# beta_zi = full._coef.detach()
def show_mses(model_perfect, model_normal):
    absc = np.arange(len(model_perfect.mse_cov_list))
    plt.plot(
        absc,
        model_perfect.mse_cov_list,
        label=r"$\|\Sigma  - \hat \Sigma \|$  init true param",
        color="blue",
    )
    plt.plot(
        absc,
        model_perfect.mse_beta_list,
        label=r"$\|B  - \hat B \|$  init true param",
        color="black",
    )
    plt.plot(
        absc,
        model_perfect.mse_infla_list,
        label=r"$\|\Theta_0  - \hat \Theta_0 \|$  init true param",
        color="red",
    )
    plt.plot(
        absc, model_perfect.mse_ksi_list, label="ksi init true param", color="green"
    )
    # plt.plot(absc, model_normal.mse_cov_list, label=r"$\|\Sigma  - \hat \Sigma \|$  init classic", color = "blue", linestyle = "--")
    # plt.plot(absc, model_normal.mse_beta_list, label=r"$\|B  - \hat B \|$  init classic", color = "black", linestyle = "--")
    # plt.plot(absc, model_normal.mse_infla_list, label=r"$\|\Theta_0  - \hat \Theta_0 \|$  init classic", color = "red", linestyle = "--")
    # plt.plot(absc, model_normal.mse_ksi_list, label="ksi init classic", linestyle = "--", color = "black")
    plt.title("Evolution de la MSE en fonction du temps et de l'initialisation.")
    plt.ylabel("MSE")
    plt.xlabel("Temps")
    # plt.ylim(1e-6)
    plt.legend()
    plt.yscale("log")
    # plt.savefig("mse_init.pdf", format = "pdf")
    plt.show()


# fig, axes = plt.subplots(2)
# sns.heatmap(full._coef.detach(), ax=axes[0])
# sns.heatmap(true_beta, ax=axes[1])
# plt.show()

# fig, axes = plt.subplots(2)
# sns.heatmap(full._coef_inflation.detach(), ax=axes[0])
# sns.heatmap(true_infla, ax=axes[1])
# plt.show()
show_mses(perfect, normal)


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
print("mean", torch.mean((perfect._latent_prob - simu[2]) ** 2))
print(
    "mean random",
    torch.mean(
        (simu[2] - torch.randint(size=(exog.shape[0], true_beta.shape[1]), high=1)) ** 2
    ),
)
