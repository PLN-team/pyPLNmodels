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
    n_samples=2000,
    zero_inflated=True,
    nb_cov=0,
    add_const=add_const,
    return_latent_variables=True,
    dim=25,
    rank=5,
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
nb_iter = 150
use_closed_form_prob = False
zi = ZIPln(
    endog,
    offsets=offsets,
    exog=exog,
    add_const=add_const,
    use_closed_form_prob=use_closed_form_prob,
    true_covariance=true_Sigma,
    true_coef=true_beta,
    true_infla=true_infla,
    ksi=ksi,
)
zi.fit(nb_max_iteration=nb_iter, tol=0)
zi.show()


def show_mses(model_perfect):
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
    plt.title("Evolution de la MSE en fonction du temps et de l'initialisation.")
    plt.ylabel("MSE")
    plt.xlabel("Temps")
    plt.ylim(1e-6)
    plt.legend()
    plt.yscale("log")
    # plt.savefig("mse_init.pdf", format = "pdf")
    plt.show()


show_mses(zi)


sns.heatmap(zi.covariance)
plt.show()
sns.heatmap(true_Sigma)
plt.show()
