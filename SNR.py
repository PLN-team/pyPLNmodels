from pyPLNmodels.models import PlnPCAcollection, PlnPCA, Pln, ZIPln
from pyPLNmodels import get_real_count_data, get_simulated_count_data


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

os.chdir("./pyPLNmodels/")


moyennes = np.linspace(-1, 4,4)

nb_iter = 150
n_samples = 100
dim = 50
rank = 10

inflated_percentages_zero_Y = []
inflated_single_percentages_zero_Y = []
inflated_n_percentages_zero_Y = []
percentages_zero_Y = []
BETA = r"$\|\hat{\Theta} - \Theta\|$"
SIGMA = r"$\|\hat{\Sigma} - \Sigma\|$"
BETA_INFLA = r"$\|\hat{\Theta}^0-\Theta^0\|$"
W = r"$\|\hat W - W\|$"
ZI_FREE = "ZIPLN without closed form"
ZI_CLOSED = "ZIPLN with closed form"
PLN = "Pln on the non inflated T"
ZI_SINGLE_FREE="ZIPLN with single inflation and no closed form"
ZI_SINGLE_CLOSED="ZIPLN with single inflation and closed form"
ZI_N_CLOSED="ZIPLN with one inflation per sample and closed form"
ZI_NP_CLOSED="ZIPLN with one inflation per entry and closed form"
ZI_N_FREE="ZIPLN with one inflation per sample and no closed form"
ZI_NP_FREE="ZIPLN with one inflation per entry and no closed form"
NEGELBO= "negative ELBO"
dict_zi_free ={ZI_FREE:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[] }}
dict_zi_single_free ={ZI_SINGLE_FREE:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[] }}
dict_zi_single_closed ={ZI_SINGLE_CLOSED:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[] }}
dict_zi_closed ={ZI_CLOSED:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_zi_n_closed ={ZI_N_CLOSED:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_zi_n_free ={ZI_N_FREE:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_zi_np_closed ={ZI_NP_CLOSED:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_zi_np_free ={ZI_NP_FREE:{BETA:[], SIGMA:[], BETA_INFLA:[], W:[],NEGELBO:[]}}
dict_pln ={PLN:{BETA:[], SIGMA:[], NEGELBO:[]}}



for moyenne in moyennes:

    add_const = False
    (
        simu,
        exog,
        offsets,
        true_Sigma,
        true_beta,
        true_infla,
    ) = get_simulated_count_data(
        return_true_param=True,
        n_samples=n_samples,
        zero_inflated=True,
        nb_cov=1,
        add_const=add_const,
        return_latent_variables=True,
        dim=dim,
        rank=rank,
        seed=0,
        to_add_coef=moyenne,
    )
    Y = simu[3]
    ksi = simu[2]
    endog_inflated = simu[0]
    endog_single_inflated = simu[4]
    endog_n_inflated = simu[5]
    ksi_single = simu[6]
    ksi_n_param = simu[7]


    inflated_percentages_zero_Y.append(
        torch.sum(endog_inflated == 0) / (endog_inflated.shape[0] * endog_inflated.shape[1])
    )
    inflated_single_percentages_zero_Y.append(
            torch.sum(endog_single_inflated == 0) / (endog_single_inflated.shape[0] * endog_single_inflated.shape[1])
            )
    inflated_n_percentages_zero_Y.append(
            torch.sum(endog_n_inflated == 0) / (endog_n_inflated.shape[0] * endog_n_inflated.shape[1])
            )
    percentages_zero_Y.append(torch.sum(Y == 0) / (Y.shape[0] * Y.shape[1]))
    percentage_zero_Y = percentages_zero_Y[-1]

    percentage_zero_endog_inflated = inflated_percentages_zero_Y[-1]
    fraction_removed = percentage_zero_endog_inflated-percentage_zero_Y
    start =int(fraction_removed*Y.shape[0])
    Y = Y[start:,:]
    offsets_Y = offsets[start:,:]
    exog_Y = exog[start:,:]



    def fit_and_append_model(model, string, is_inflated, dict_model):
        model.fit(nb_max_iteration=nb_iter, tol=0)
        dict_model[string][BETA].append(model.mse_beta_list[-1])
        dict_model[string][SIGMA].append(model.mse_cov_list[-1])
        dict_model[string][NEGELBO].append(-model._elbos_list[-1])
        if is_inflated:
            dict_model[string][BETA_INFLA].append(model.mse_infla_list[-1])
            dict_model[string][W].append(model.mse_ksi_list[-1])
        return model



    zi_n_free = ZIPln(
        endog_n_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=False,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi_n_param,
        do_n_inflation = True
    )
    zi_n_free = fit_and_append_model(zi_n_free, ZI_N_FREE, True, dict_zi_n_free)


    zi_np_free = ZIPln(
        endog_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=False,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
        do_np_inflation = True
    )
    zi_np_free = fit_and_append_model(zi_np_free, ZI_NP_FREE, True, dict_zi_np_free)

    zi_n_closed_form = ZIPln(
        endog_n_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=True,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi_n_param,
        do_n_inflation = True
    )
    zi_n_closed_form = fit_and_append_model(zi_n_closed_form, ZI_N_CLOSED, True, dict_zi_n_closed)


    zi_np_closed_form = ZIPln(
        endog_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=True,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
        do_np_inflation = True
    )
    zi_np_closed_form = fit_and_append_model(zi_np_closed_form, ZI_NP_CLOSED, True, dict_zi_np_closed)


    zi_single_closed_form = ZIPln(
        endog_single_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=True,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi_single,
        do_single_inflation = True
    )
    zi_single_closed_form = fit_and_append_model(zi_single_closed_form, ZI_SINGLE_CLOSED, True, dict_zi_single_closed)

    zi_single_free_form = ZIPln(
        endog_single_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=False,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi_single,
        do_single_inflation = True
    )
    zi_single_free_form = fit_and_append_model(zi_single_free_form, ZI_SINGLE_FREE, True, dict_zi_single_free)

    zi_closed_form = ZIPln(
        endog_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=True,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
    )
    zi_closed_form = fit_and_append_model(zi_closed_form, ZI_CLOSED, True, dict_zi_closed)
    ref_neg_elbo = -zi_closed_form._elbos_list[-1]

    zi_free = ZIPln(
        endog_inflated,
        offsets=offsets,
        exog=exog,
        add_const=add_const,
        use_closed_form_prob=False,
        true_covariance=true_Sigma,
        true_coef=true_beta,
        true_infla=true_infla,
        ksi=ksi,
    )
    fit_and_append_model(zi_free, ZI_FREE, True, dict_zi_free)

    pln = Pln(
        Y,
        exog = exog_Y,
        offsets=offsets_Y,
        add_const=add_const,
        true_covariance=true_Sigma,
        true_coef=true_beta,
    )
    fit_and_append_model(pln, PLN, False, dict_pln)

fig, axes = plt.subplots(3,2, figsize = (15,15))

ax_mse_sigma = axes[0,0]
ax_mse_beta = axes[1,0]
ax_mse_beta_infla = axes[2,0]
ax_elbo = axes[0,1]
ax_percentages = axes[1,1]
ax_mse_ksi = axes[2,1]

def plot_model(dict_model, color, linestyle = "solid"):
    model_name = list(dict_model.keys())[0]
    true_dict = dict_model[model_name]
    ax_mse_beta.plot(moyennes, true_dict[BETA], color = color, linestyle = linestyle)
    ax_mse_sigma.plot(moyennes, true_dict[SIGMA], color = color, linestyle = linestyle, label = model_name)
    ax_elbo.plot(moyennes, true_dict[NEGELBO], color = color, linestyle = linestyle)
    if model_name != PLN:
        ax_mse_beta_infla.plot(moyennes, true_dict[BETA_INFLA], linestyle = linestyle, color = color)
        ax_mse_ksi.plot(moyennes, true_dict[W], linestyle = linestyle, color = color)

plot_model(dict_zi_closed, "red")
plot_model(dict_zi_free, "red", linestyle = "--")
plot_model(dict_zi_single_closed, "green")
plot_model(dict_zi_single_free,"green", linestyle = '--' )
plot_model(dict_zi_n_closed,"black")
plot_model(dict_zi_n_free,"black", linestyle = '--')
plot_model(dict_zi_np_closed,"orange")
plot_model(dict_zi_np_free,"orange", linestyle = '--' )
plot_model(dict_pln, "blue", linestyle = "dotted")


ax_percentages.plot(moyennes, percentages_zero_Y, label = "Percentages of zero in T", color = "blue")
ax_percentages.plot(moyennes, inflated_percentages_zero_Y, label = "Percentages of zero in Y (inflated)", color = "red")
ax_percentages.plot(moyennes, inflated_single_percentages_zero_Y, label = "Percentages of zero in Y (single inflated)", color = "green")
ax_percentages.plot(moyennes, inflated_n_percentages_zero_Y, label = "Percentages of zero in Y (inflated per sample)", color = "black")


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

handles, labels = ax_mse_sigma.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol = 3)

plt.savefig("zi_stats.pdf", format = "pdf")
plt.show()

