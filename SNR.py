from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")


moyennes = np.linspace(-3, 6, 30)


inflated_percentages_zero_Y = []
percentages_zero_Y = []
BETA = r"$\|\hat{\Theta} - \Theta\|$"
SIGMA = r"$\|\hat{\Sigma} - \Sigma\|$"
BETA_INFLA = r"$\|\hat{\Theta}^0-\Theta^0\|$"
W = r"$\|\hat W - W\|$"
ZI_FREE = "ZIPLN without closed form"
ZI_CLOSED = "ZIPLN with closed form"
PLN = "Pln on the non inflated T"
NEGELBO= "negative ELBO"
dict_zi_free ={ZI_FREE:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[] }}
dict_zi_closed ={ZI_CLOSED:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_pln ={PLN:{BETA:[], SIGMA:[], NEGELBO:[]}}



for moyenne in moyennes:

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
        n_samples=500,
        zero_inflated=True,
        nb_cov=0,
        add_const=add_const,
        return_latent_variables=True,
        dim=50,
        rank=10,
        seed=0,
        to_add_coef=moyenne,
    )
    Y = simu[3]
    ksi = simu[2]
    endog = simu[0]

    inflated_percentages_zero_Y.append(
        torch.sum(endog == 0) / (endog.shape[0] * endog.shape[1])
    )
    percentages_zero_Y.append(torch.sum(Y == 0) / (Y.shape[0] * Y.shape[1]))

    nb_iter = 1000
    zi_closed_form = ZIPln(
        endog,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=True,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
    )
    zi_closed_form.fit(nb_max_iteration=nb_iter, tol=0)
    dict_zi_closed[ZI_CLOSED][BETA].append(zi_closed_form.mse_beta_list[-1])
    dict_zi_closed[ZI_CLOSED][SIGMA].append(zi_closed_form.mse_cov_list[-1])
    dict_zi_closed[ZI_CLOSED][BETA_INFLA].append(zi_closed_form.mse_infla_list[-1])
    dict_zi_closed[ZI_CLOSED][W].append(zi_closed_form.mse_ksi_list[-1])
    dict_zi_closed[ZI_CLOSED][NEGELBO].append(-zi_closed_form._elbos_list[-1])
    ref_neg_elbo = -zi_closed_form._elbos_list[-1]

    zi_free = ZIPln(
        endog,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=False,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
    )
    zi_free.fit(nb_max_iteration=nb_iter, tol=0)
    dict_zi_free[ZI_FREE][BETA].append(zi_free.mse_beta_list[-1])
    dict_zi_free[ZI_FREE][SIGMA].append(zi_free.mse_cov_list[-1])
    dict_zi_free[ZI_FREE][BETA_INFLA].append(zi_free.mse_infla_list[-1])
    dict_zi_free[ZI_FREE][W].append(zi_free.mse_ksi_list[-1])
    dict_zi_free[ZI_FREE][NEGELBO].append(-zi_free._elbos_list[-1] - ref_neg_elbo)

    pln = Pln(
        Y,
        offsets=offsets,
        add_const=add_const,
        true_covariance=true_Sigma,
        true_coef=true_beta,
    )
    pln.fit(nb_max_iteration = nb_iter, tol = 0)
    dict_pln[PLN][BETA].append(pln.mse_beta_list[-1])
    dict_pln[PLN][SIGMA].append(pln.mse_cov_list[-1])
    dict_pln[PLN][NEGELBO].append(-pln._elbos_list[-1] - ref_neg_elbo)

fig, axes = plt.subplots(6,1, figsize = (15,15))

ax_mse_sigma = axes[0]
ax_mse_beta = axes[1]
ax_mse_beta_infla = axes[2]
ax_mse_ksi = axes[3]
ax_elbo = axes[4]
ax_percentages = axes[5]
def plot_model(dict_model):
    model_name = list(dict_model.keys())[0]
    true_dict = dict_model[model_name]
    ax_mse_beta.plot(moyennes, true_dict[BETA], label =model_name)
    ax_mse_sigma.plot(moyennes, true_dict[SIGMA], label =model_name)
    ax_elbo.plot(moyennes, true_dict[NEGELBO], label =model_name)
    if model_name != PLN:
        ax_mse_beta_infla.plot(moyennes, true_dict[BETA_INFLA], label = model_name)
        ax_mse_ksi.plot(moyennes, true_dict[W], label = model_name)

plot_model(dict_zi_closed)
plot_model(dict_zi_free)
plot_model(dict_pln)


ax_percentages.plot(moyennes, percentages_zero_Y, label = "Percentages of zero in T", color = "red")
ax_percentages.plot(moyennes, inflated_percentages_zero_Y, label = "Percentages of zero in Y (inflated)", color = "black")


xlabel = r"$E[Z_{ij}]$"

ax_mse_beta.set_ylabel(BETA, rotation = "horizontal")
ax_mse_beta.set_xlabel(xlabel)
ax_mse_beta.set_yscale("log")
ax_mse_beta_infla.set_ylabel(BETA_INFLA, rotation = "horizontal")
ax_mse_beta_infla.set_xlabel(xlabel)
ax_mse_beta_infla.set_yscale("log")
ax_mse_sigma.set_ylabel(SIGMA, rotation = "horizontal")
ax_mse_sigma.set_xlabel(xlabel)
ax_mse_sigma.set_yscale("log")
ax_mse_ksi.set_ylabel(W, rotation = "horizontal")
ax_mse_ksi.set_xlabel(xlabel)
ax_mse_ksi.set_yscale("log")
# ax_elbo.set_yscale("log")
ax_elbo.set_ylabel(NEGELBO, rotation = "horizontal")
ax_elbo.set_xlabel(xlabel)




ax_percentages.set_xlabel(xlabel)
ax_percentages.set_ylabel("Percentage of zeros", rotation = "horizontal")
ax_percentages.set_yscale("log")

ax_percentages.legend()
ax_mse_beta.legend()
ax_mse_sigma.legend()
ax_mse_ksi.legend()
ax_mse_beta_infla.legend()
ax_elbo.legend()
plt.savefig("zi_stats.pdf", format = "pdf")
plt.show()

