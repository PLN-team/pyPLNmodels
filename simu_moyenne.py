from os.path import exists
from tqdm import tqdm
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

from pyPLNmodels import (
    # get_real_count_data,
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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = "cpu"

viz = "poisson"
_moyennes_XB = np.linspace(0, 3, 7)
_moyennes_proba = np.linspace(0.1, 0.8, 7)
# chosen_moyennes = [_moyennes_XB[0], _moyennes_XB[3], _moyennes_XB[6], _moyennes_XB[9], _moyennes_XB[12], _moyennes_XB[14]]
_mean_infla = 0.30
_mean_xb = 2

_nb_bootstrap = 15


if viz == "poisson":
    _moyennes = _moyennes_XB
    _mean_sim = _mean_infla
elif viz == "proba":
    _moyennes = _moyennes_proba
    _mean_sim = _mean_xb


chosen_moyennes = _moyennes[:-1]

n = 350
dim = 100
inflation_formula = "column-wise"


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
B_KEY = "RMSE_B"
SIGMA_KEY = "RMSE_SIGMA"
PI_KEY = "RMSE_PI"
ELBO_KEY = "ELBO"
B0_KEY = "RMSE_B0"

VIZ_LABEL = {"proba": r"$\pi$", "poisson": r"$XB$"}


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
    B_KEY: r"RMSE($\hat{\beta} - \beta^{\bigstar}$)",
    SIGMA_KEY: r"RMSE($\hat{\Omega} - \Omega^{\bigstar}$)",
    B0_KEY: r"RMSE($\hat{B}^0 - B^{0\bigstar}$)",
    ELBO_KEY: "Negative ELBO (Lower the better)",
    PI_KEY: r"RMSE($\hat{\pi} - \pi^{\bigstar}$)",
}
CRITERION_KEYS = [ELBO_KEY, REC_KEY, SIGMA_KEY, B_KEY, PI_KEY, B0_KEY]
COLORS = {
    ENH_FREE_KEY: "cornflowerblue",
    ENH_CLOSED_KEY: "darkblue",
    STD_FREE_KEY: "lightcoral",
    STD_CLOSED_KEY: "darkred",
}


KEY_MODELS = [ENH_CLOSED_KEY, STD_FREE_KEY, ENH_FREE_KEY, STD_CLOSED_KEY]


def RMSE(t):
    return torch.sqrt(torch.mean(t**2))


def get_dict_models(endog, exog, exog_inflation, offsets, inflation_formula):
    sim_models = {
        ENH_FREE_KEY: ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=True,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
            use_closed_form_prob=False,
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
            use_closed_form_prob=False,
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


def get_plnparam(inflation_formula):
    if inflation_formula == "global":
        nb_cov_infla = 0
        add_const_inflation = False
    else:
        nb_cov_infla = 0
        add_const_inflation = True
    plnparam = get_simulation_parameters(
        nb_cov=1,
        add_const=True,
        nb_cov_inflation=nb_cov_infla,
        zero_inflation_formula=inflation_formula,
        n_samples=n,
        add_const_inflation=add_const_inflation,
        dim=dim,
    )
    plnparam._offsets *= 0
    return plnparam


def fit_models(dict_models):
    for key, model in dict_models.items():
        # model.fit(tol = 0, nb_max_iteration=1500)
        model.fit(tol=0, nb_max_iteration=1000)
    return dict_models


class one_plot:
    def __init__(
        self,
        moyennes,
        mean_XB_or_prob,
        chosen_moyennes,
        nb_bootstrap,
        inflation_formula,
        viz,
    ):
        self.moyennes = moyennes
        self.chosen_moyennes = chosen_moyennes
        self.nb_bootstrap = nb_bootstrap
        self.inflation_formula = inflation_formula
        if viz not in ["poisson", "proba"]:
            raise ValueError("Wrong visualization")
        if viz == "proba":
            for moyenne in moyennes:
                if moyenne < 0 or moyenne > 1:
                    raise ValueError("Wrong viz for the moyenne")
        if viz == "poisson":
            self.mean_infla = mean_XB_or_prob
        else:
            self.mean_xb = mean_XB_or_prob

        self.viz = viz
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
                for moyenne in self.moyennes
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
        samples_only_zeros = torch.sum(endog, axis=1) == 0
        dim_only_zeros = torch.sum(endog, axis=0) == 0
        for model in dict_models.values():
            model.samples_only_zeros = samples_only_zeros
            model.dim_only_zeros = dim_only_zeros

        # sns.heatmap(torch.inverse(_plnparam.covariance))
        # plt.title("True")
        # plt.show()
        fit_models(dict_models)
        return dict_models

    def simulate(self):
        if exists(self.abs_name_file):
            print("Getting back data")
            with open(self.abs_name_file, "rb") as fp:
                self.model_criterions = pickle.load(fp)
        else:
            print("Simulating")
            plnparam = get_plnparam(self.inflation_formula)
            if self.viz == "poisson":
                plnparam._set_mean_proba(self.mean_infla)
            else:
                plnparam._set_gaussian_mean(self.mean_xb)
            Sigma = plnparam.covariance
            for moyenne in tqdm(self.moyennes):
                if self.viz == "poisson":
                    plnparam._set_gaussian_mean(moyenne)
                    print("true mean gaussian", moyenne)
                    print("true mean proba", self.mean_infla)
                else:
                    plnparam._set_mean_proba(moyenne)
                    print("true mean gaussian", self.mean_xb)
                    print("true mean proba", moyenne)
                B = plnparam.coef
                B0 = plnparam.coef_inflation
                print("mean gaussian", torch.mean(plnparam.gaussian_mean))
                print("mean proba", torch.mean(plnparam.proba_inflation))
                for i in range(self.nb_bootstrap):
                    dict_models = self.simulate_mean(plnparam, i)
                    self.stock_results(dict_models, moyenne, Sigma, B, B0)
            self.save_criterions()

    def stock_results(self, dict_models, moyenne, Sigma, B, B0):
        for key_model in KEY_MODELS:
            model_fitted = dict_models[key_model]
            lines = ~model_fitted.samples_only_zeros
            cols = ~model_fitted.dim_only_zeros
            omega = torch.inverse(Sigma)[cols, cols]
            beta = B[:, cols]
            if model_fitted._zero_inflation_formula == "row-wise":
                beta_0 = B0[lines, :]
            elif model_fitted._zero_inflation_formula == "column-wise":
                beta_0 = B0[:, cols]
            else:
                beta_0 = B0
            results_model = self.model_criterions[key_model][moyenne]
            results_model[REC_KEY].append(model_fitted.reconstruction_error)
            results_model[SIGMA_KEY].append(
                RMSE(torch.inverse(model_fitted.covariance) - omega.cpu())
            )
            results_model[B_KEY].append(RMSE(model_fitted.coef - beta.cpu()))
            results_model[PI_KEY].append(
                RMSE(
                    torch.sigmoid(model_fitted.coef_inflation)
                    - torch.sigmoid(beta_0).cpu()
                )
            )
            results_model[ELBO_KEY].append(model_fitted.elbo)
            results_model[B0_KEY].append(
                RMSE(model_fitted.coef_inflation - beta_0.cpu())
            )

    def save_criterions(self):
        with open(self.abs_name_file, "wb") as fp:
            pickle.dump(self.model_criterions, fp)

    @property
    def abs_name_file(self):
        return f"results_simu/{self.name_file}"

    @property
    def name_file(self):
        return (
            f"{self.moyennes[0]}_{self.moyennes[len(self.moyennes)-1]}_{len(self.moyennes)}"
            + str(self.nb_bootstrap)
            + self.viz
            + self.inflation_formula
            + f"n_{n}_dim_{dim}"
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
        data = self.data
        data.to_csv("df_simu.csv")
        for crit_key in CRITERION_KEYS:
            palette = {
                LABEL_DICT[model_key]: COLORS[model_key] for model_key in KEY_MODELS
            }
            data["moyenne"] = np.round(data["moyenne"], 2)
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
                ax.set_xlabel(VIZ_LABEL[self.viz], fontsize=22)

        # for crit_key in CRITERION_KEYS:
        # plots[B0_KEY].tick_params(axis = "y", labelsize = 18)
        # plots["ELBO"].tick_params(axis = "both", labelsize = 1)
        ax = plots[B0_KEY]
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend()
        ax.legend(
            handles=[handles[0], handles[1], handles[2], handles[3]],
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=12,
        )
        ax_b = plots[B_KEY]
        ax_y = plots[REC_KEY]
        ax_y.set_ylim(0.1, 10)
        ax_b.set_ylim(10 ** (-2), 1)
        if viz == "poisson":
            title = rf"n={n},p={dim},d=1,$\pi \approx {_mean_infla}$_{self.inflation_formula}"
        else:
            title = (
                rf"n={n},p={dim},d=1,$XB \approx {_mean_xb}$_{self.inflation_formula}"
            )
        fig.suptitle(title)
        doss_viz = "poisson_viz" if self.viz == "poisson" else "proba_viz"
        plt.savefig(
            f"figures/{doss_viz}/simu{self.inflation_formula}_{self.name_file}.png",
            format="png",
        )
        # plt.show()


def launch_formulas(_moyennes, _mean_sim, chosen_moyennes, _nb_bootstrap, viz):
    for inflation_formula in tqdm(["row-wise", "column-wise", "global"]):
        op = one_plot(
            _moyennes,
            _mean_sim,
            chosen_moyennes,
            _nb_bootstrap,
            inflation_formula,
            viz=viz,
        )
        op.simulate()
        op.plot_results()
    plt.show()


launch_formulas(_moyennes, _mean_sim, chosen_moyennes, _nb_bootstrap, viz)
