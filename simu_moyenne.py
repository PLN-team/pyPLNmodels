from os.path import exists
from tqdm import tqdm
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

from pyPLNmodels import (
    get_real_count_data,
    ZIPln,
    Pln,
    get_zipln_simulated_count_data,
    get_simulation_parameters,
    sample_zipln,
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

_moyennes_XB = np.linspace(4, 5, 2)
# chosen_moyennes = [_moyennes_XB[0], _moyennes_XB[3], _moyennes_XB[6], _moyennes_XB[9], _moyennes_XB[12], _moyennes_XB[14]]
chosen_moyennes = _moyennes_XB

_mean_infla = 0.22
_nb_bootstrap = 1

n = 350
dim = 301
inflation_formula = "row-wise"
title = rf"n={n},p={dim},d=1,$\pi \approx {_mean_infla}$"

import pickle

ENH_CLOSED_KEY = "enhanced_closed"
ENH_FREE_KEY = "enhanced_free"
STD_CLOSED_KEY = "standard_closed"
STD_FREE_KEY = "standard_free"


LABEL_DICT = {
    ENH_CLOSED_KEY: "Enhanced Analytic",
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


# LEGEND_DICT = {
#     REC_KEY: "Reconstruction_error",
#     B_KEY: r"$\|\|\hat{\beta} - \beta\| \|^2_2$",
#     SIGMA_KEY: r"$\|\|\hat{\Sigma} - \Sigma\|\|^2_2$",
#     B0_KEY: r"$\|\|\hat{B}^0 - B^0\|\|^2_2$",
#     ELBO_KEY: "Negative ELBO (Lower the better)",
#     PI_KEY: r"$\|\|\hat{\pi} - \pi\|\|_1$",
# }

LEGEND_DICT = {
    REC_KEY: "Reconstruction_error",
    B_KEY: r"MSE($\hat{\beta} - \beta^{\bigstar}$)",
    SIGMA_KEY: r"MSE($\hat{\Sigma} - \Sigma^{\bigstar}$)",
    B0_KEY: r"MSE($\hat{B}^0 - B^{0\bigstar}$)",
    ELBO_KEY: "Negative ELBO (Lower the better)",
    PI_KEY: r"MAE($\hat{\pi} - \pi^{\bigstar}$)",
}
CRITERION_KEYS = [ELBO_KEY, REC_KEY, SIGMA_KEY, B_KEY, PI_KEY, B0_KEY]
COLORS = {
    ENH_FREE_KEY: "cornflowerblue",
    ENH_CLOSED_KEY: "darkblue",
    STD_FREE_KEY: "lightcoral",
    STD_CLOSED_KEY: "darkred",
}


KEY_MODELS = [ENH_CLOSED_KEY, ENH_FREE_KEY, STD_FREE_KEY, STD_CLOSED_KEY]


def MSE(t):
    return torch.mean(t**2)


def MAE(t):
    return torch.mean(torch.abs(t))


def get_dict_models(endog, exog, exog_inflation, offsets, inflation_formula):
    sim_models = {
        ENH_FREE_KEY: ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
        ENH_CLOSED_KEY: ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            use_closed_form_prob=True,
            add_const_inflation=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
        STD_FREE_KEY: Brute_ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
        STD_CLOSED_KEY: Brute_ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            use_closed_form_prob=True,
            add_const_inflation=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
    }
    return sim_models


def get_plnparam(mean_xb, mean_infla, inflation_formula):
    if inflation_formula == "global":
        nb_cov = 0
        add_const_inflation = False
    else:
        nb_cov = 2
        add_const_inflation = True
    plnparam = get_simulation_parameters(
        add_const=True,
        nb_cov_inflation=nb_cov,
        zero_inflation_formula=inflation_formula,
        n_samples=n,
        add_const_inflation=add_const_inflation,
    )
    plnparam._coef += mean_xb - torch.mean(plnparam._coef)
    plnparam._coef_inflation += logit(mean_infla) - logit(
        torch.mean(torch.sigmoid(plnparam._coef_inflation)).cpu()
    )

    plnparam._offsets *= 0
    return plnparam


def fit_models(dict_models):
    for key, model in dict_models.items():
        model.fit()
    return dict_models


class one_plot:
    def __init__(
        self, moyennes_XB, mean_infla, chosen_moyennes, nb_bootstrap, inflation_formula
    ):
        self.moyennes_XB = moyennes_XB
        self.chosen_moyennes = chosen_moyennes
        self.mean_infla = mean_infla
        self.nb_bootstrap = nb_bootstrap
        self.inflation_formula = inflation_formula
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
        endog = sample_zipln(_plnparam, seed=i)
        dict_models = get_dict_models(
            endog,
            exog=_plnparam.exog,
            offsets=_plnparam.offsets,
            exog_inflation=_plnparam.exog_inflation,
            inflation_formula=self.inflation_formula,
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
                plnparam = get_plnparam(
                    moyenne, self.mean_infla, self.inflation_formula
                )
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
            print("first", model_fitted.coef_inflation)
            print("other", B0)
            x
            results_model[B0_KEY].append(MSE(model_fitted.coef_inflation - B0.cpu()))

    def save_criterions(self):
        with open(self.abs_name_file, "wb") as fp:
            pickle.dump(self.model_criterions, fp)

    @property
    def abs_name_file(self):
        return f"results_simu/{self.name_file}"

    @property
    def name_file(self):
        return (
            str(self.moyennes_XB)
            + str(self.nb_bootstrap)
            + str(self.mean_infla)
            + self.inflation_formula
        )

    @property
    def data(self):
        columns = ["model_name", "moyenne"]
        columns += CRITERION_KEYS
        data = pd.DataFrame(columns=columns)
        for model_key in KEY_MODELS:
            for moyenne in self.chosen_moyennes:
                for i in range(self.nb_bootstrap):
                    values = {
                        "model_name": [LABEL_DICT[model_key]],
                        "moyenne": [moyenne],
                    }
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
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        plots = {}
        plots[REC_KEY] = axes[1, 1]
        plots[REC_KEY].set_title("Reconstruction error", fontsize=22)
        plots[SIGMA_KEY] = axes[0, 1]
        plots[SIGMA_KEY].set_title(LEGEND_DICT[SIGMA_KEY], fontsize=22)
        plots[B_KEY] = axes[0, 2]
        plots[B_KEY].set_title(LEGEND_DICT[B_KEY], fontsize=22)
        plots[PI_KEY] = axes[1, 2]
        plots[PI_KEY].set_title(LEGEND_DICT[PI_KEY], fontsize=22)
        plots[B0_KEY] = axes[0, 0]
        plots[B0_KEY].set_title(LEGEND_DICT[B0_KEY], fontsize=22)
        plots[ELBO_KEY] = axes[1, 0]
        plots[ELBO_KEY].set_title(LEGEND_DICT[ELBO_KEY], fontsize=22)
        for key, plot in plots.items():
            if key != ELBO_KEY:
                plot.set_yscale("log")
            else:
                pass
                # plot.set_yscale("symlog")
        # for axe in axes:
        #     for ax in axe:
        #         ax.set_yscale("log")
        # plots[ELBO_KEY].set_yscale("symlog")
        data = self.data
        for crit_key in CRITERION_KEYS:
            palette = {
                LABEL_DICT[model_key]: COLORS[model_key] for model_key in KEY_MODELS
            }
            data["moyenne"] = np.round(data["moyenne"], 2)
            # if crit_key != "ELBO":
            sns.boxplot(
                data=data,
                x="moyenne",
                y=crit_key,
                hue="model_name",
                ax=plots[crit_key],
                palette=palette,
                boxprops={"alpha": 0.4},
            )
            sns.stripplot(
                data=data,
                x="moyenne",
                y=crit_key,
                hue="model_name",
                dodge=True,
                ax=plots[crit_key],
                palette=palette,
            )
            plots[crit_key].tick_params(axis="y", labelsize=22)
        for axe in axes:
            for ax in axe:
                ax.tick_params(axis="both", labelsize=22)
                ax.set_ylabel("")
                ax.legend([], [], frameon=False)
                ax.set_xlabel(r"Mean $XB$", fontsize=22)

        # for crit_key in CRITERION_KEYS:
        # plots[B0_KEY].tick_params(axis = "y", labelsize = 18)
        # plots["ELBO"].tick_params(axis = "both", labelsize = 1)
        ax = plots[B0_KEY]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=[handles[0], handles[2], handles[1], handles[3]],
            labels=LABEL_DICT.values(),
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=12,
        )
        plt.savefig("simu.png", format="png")
        plt.show()


op = one_plot(
    _moyennes_XB, _mean_infla, chosen_moyennes, _nb_bootstrap, inflation_formula
)
op.simulate()
op.plot_results()
