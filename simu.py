from os.path import exists
from tqdm import tqdm
import seaborn as sns

from pyPLNmodels import (
    get_real_count_data,
    ZIPln,
    Pln,
    get_simulated_count_data,
    get_simulation_parameters,
    sample_pln,
)
from pyPLNmodels.models import Brute_ZIPln
import matplotlib.pyplot as plt
from scipy.special import logit
import scipy.stats as ss
import numpy as np
import pandas as pd
import torch
import math
from matplotlib.ticker import FormatStrFormatter


import pickle

ENH_CLOSED_KEY = "enhanced_closed"
ENH_FREE_KEY = "enhanced_free"
STD_CLOSED_KEY = "standard_closed"
STD_FREE_KEY = "standard_free"


LABEL_DICT = {
    ENH_CLOSED_KEY: "Enhanced analytic",
    ENH_FREE_KEY: "Enhanced",
    STD_CLOSED_KEY: "Standard Analytic",
    STD_FREE_KEY: "Standard",
}

REC_KEY = "Reconstruction_error"
B_KEY = "MSE_B"
SIGMA_KEY = "MSE_SIGMA"
PI_KEY = "MAE_PI"
ELBO_KEY = "ELBO"
B0_KEY = "MSE_B0"


LEGEND_DICT = {
    REC_KEY: "Reconstruction_error",
    B_KEY: r"$\|\hat{\beta} - \beta \|^2_2$",
    SIGMA_KEY: r"$\|\hat{\Sigma} - \Sigma\|^2_2$",
    B0_KEY: r"$\|\hat{B}^0 - B^0\|^2_2$",
    ELBO_KEY: "Negative ELBO (Lower the better)",
    PI_KEY: r"$\|\hat{\pi} - \pi\|_1$",
}

CRITERION_KEYS = [REC_KEY, SIGMA_KEY, B_KEY, PI_KEY, ELBO_KEY, B0_KEY]
COLORS = {
    ENH_FREE_KEY: "cornflowerblue",
    ENH_CLOSED_KEY: "darkblue",
    STD_CLOSED_KEY: "darkred",
    STD_FREE_KEY: "lightcoral",
}

_moyennes_XB = np.linspace(0, 6, 6)
# chosen_moyennes = [_moyennes_XB[0], _moyennes_XB[3], _moyennes_XB[6], _moyennes_XB[9], _moyennes_XB[12], _moyennes_XB[14]]
chosen_moyennes = _moyennes_XB

_mean_infla = 0.2
_nb_bootstrap = 8


KEY_MODELS = [ENH_CLOSED_KEY, ENH_FREE_KEY, STD_FREE_KEY, STD_CLOSED_KEY]


def MSE(t):
    return torch.mean(t**2)


def MAE(t):
    return torch.mean(torch.abs(t))


def get_dict_models(endog, exog, offsets):
    sim_models = {
        ENH_FREE_KEY: ZIPln(endog, exog=exog, offsets=offsets),
        ENH_CLOSED_KEY: ZIPln(
            endog, exog=exog, offsets=offsets, use_closed_form_prob=True
        ),
        STD_FREE_KEY: Brute_ZIPln(endog, exog=exog, offsets=offsets),
        STD_CLOSED_KEY: Brute_ZIPln(
            endog, exog=exog, offsets=offsets, use_closed_form_prob=True
        ),
    }
    return sim_models


n = 100
dim = 25
title = rf"n={n},p={dim},d=1,$\pi \approx {_mean_infla}$"


def get_plnparam(mean_xb, mean_infla):
    plnparam = get_simulation_parameters(
        add_const=True, nb_cov=0, zero_inflated=True, n_samples=n
    )
    plnparam._coef += mean_xb - torch.mean(plnparam._coef)
    plnparam._coef_inflation += logit(mean_infla) - logit(
        torch.mean(torch.sigmoid(plnparam._coef_inflation)).cpu()
    )
    # plnparam._offsets *=0
    return plnparam


def get_data(_plnparam, seed):
    endog = sample_pln(_plnparam, seed=seed)
    print("XB", torch.mean(_plnparam.exog @ _plnparam.coef))
    print("pi", torch.mean(torch.sigmoid(_plnparam.exog @ _plnparam.coef_inflation)))


def fit_models(dict_models):
    for key, model in dict_models.items():
        model.fit()
    return dict_models


class one_plot:
    def __init__(self, moyennes_XB, mean_infla, chosen_moyennes, nb_bootstrap):
        self.moyennes_XB = moyennes_XB
        self.chosen_moyennes = chosen_moyennes
        self.mean_infla = mean_infla
        self.nb_bootstrap = nb_bootstrap
        self.model_criterions = {
            key_model: {
                moyenne: {
                    REC_KEY: [],
                    SIGMA_KEY: [],
                    B_KEY: [],
                    PI_KEY: [],
                    ELBO_KEY: [],
                    B0_KEY: [],
                }
                for moyenne in self.moyennes_XB
            }
            for key_model in KEY_MODELS
        }

    def simulate_mean(self, _plnparam, i):
        endog = sample_pln(_plnparam, seed=i)
        dict_models = get_dict_models(
            endog, exog=_plnparam.exog, offsets=_plnparam.offsets
        )
        fit_models(dict_models)
        return dict_models

    def simulate(self):
        if exists(self.abs_name_file):
            print("Getting back data")
            with open(self.abs_name_file, "rb") as fp:
                self.model_criterions = pickle.load(fp)
        else:
            print("Simulating")
            for moyenne in tqdm(self.moyennes_XB):
                plnparam = get_plnparam(moyenne, self.mean_infla)
                Sigma = plnparam.covariance
                B = plnparam.coef
                B0 = plnparam.coef_inflation
                for i in range(self.nb_bootstrap):
                    dict_models = self.simulate_mean(plnparam, i)
                    self.stock_results(dict_models, moyenne, Sigma, B, B0)
            self.save_criterions()

    def stock_results(self, dict_models, moyenne, Sigma, B, B0):
        for key_model in KEY_MODELS:
            model_fitted = dict_models[key_model]
            results_model = self.model_criterions[key_model][moyenne]
            results_model[REC_KEY].append(model_fitted.reconstruction_error)
            results_model[SIGMA_KEY].append(MSE(model_fitted.covariance - Sigma.cpu()))
            results_model[B_KEY].append(MSE(model_fitted.coef - B.cpu()))
            results_model[PI_KEY].append(
                MAE(
                    torch.sigmoid(model_fitted.coef_inflation) - torch.sigmoid(B0).cpu()
                )
            )
            results_model[ELBO_KEY].append(model_fitted.elbo)
            results_model[B0_KEY].append(MSE(model_fitted.coef_inflation - B0.cpu()))

    def save_criterions(self):
        with open(self.abs_name_file, "wb") as fp:
            pickle.dump(self.model_criterions, fp)

    @property
    def abs_name_file(self):
        return f"results_simu/{self.name_file}"

    @property
    def name_file(self):
        return str(self.moyennes_XB) + str(self.nb_bootstrap) + str(self.mean_infla)

    @property
    def data(self):
        columns = ["model_name", "moyenne"]
        columns += CRITERION_KEYS
        data = pd.DataFrame(columns=columns)
        for model_key in KEY_MODELS:
            for moyenne in self.chosen_moyennes:
                for i in range(self.nb_bootstrap):
                    values = {"model_name": [model_key], "moyenne": [moyenne]}
                    for crit_key in CRITERION_KEYS:
                        value = self.model_criterions[model_key][moyenne][crit_key][i]
                        if crit_key == ELBO_KEY:
                            value = -value
                        if isinstance(value, torch.Tensor):
                            values[crit_key] = [value.item()]
                        else:
                            values[crit_key] = [value]
                    new_line = pd.DataFrame(values)
                    data = pd.concat((data, new_line))
        return data

    def plot_results(self):
        fig, axes = plt.subplots(2, 3, figsize=(20, 20))
        plots = {}
        plots[REC_KEY] = axes[0, 0]
        plots[REC_KEY].set_title("Reconstruction error")
        plots[SIGMA_KEY] = axes[0, 1]
        plots[SIGMA_KEY].set_title(LEGEND_DICT[SIGMA_KEY])
        plots[B_KEY] = axes[0, 2]
        plots[B_KEY].set_title(LEGEND_DICT[B_KEY])
        plots[PI_KEY] = axes[1, 0]
        plots[PI_KEY].set_title(LEGEND_DICT[PI_KEY])
        plots[B0_KEY] = axes[1, 1]
        plots[B0_KEY].set_title(LEGEND_DICT[B0_KEY])
        plots[ELBO_KEY] = axes[1, 2]
        plots[ELBO_KEY].set_title(LEGEND_DICT[ELBO_KEY])
        for axe in axes:
            for ax in axe:
                ax.set_yscale("log")
        data = self.data
        for crit_key in CRITERION_KEYS:
            sns.boxplot(
                data=data, x="moyenne", y=crit_key, hue="model_name", ax=plots[crit_key]
            )
        for axe in axes:
            for ax in axe:
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.legend([], [], frameon=False)
                ax.set_xlabel(r"Moyenne $XB$")

        plots[REC_KEY].legend()
        fig.suptitle(title)
        plt.savefig("simu.pdf", format="pdf")
        plt.show()


op = one_plot(_moyennes_XB, _mean_infla, chosen_moyennes, _nb_bootstrap)
op.simulate()
op.plot_results()
