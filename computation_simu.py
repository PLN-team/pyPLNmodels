from os.path import exists

import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import math

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

mean_poiss = 2
mean_infla = 0.3
# ns = np.linspace(100, 400, 7)
# p = 150
ps = np.linspace(100, 500, 9)
n = 500
nb_cov = 2
nb_cov_infla = 2
good_fit = True  ## good_fit is actually 1000
viz = "dims"
nb_bootstrap = 30


# mean_poiss = 2
# mean_infla = 0.3
# # ns = np.linspace(100, 300, 2)
# ps = np.linspace(100, 300, 2)
# n = 175
# # p = 175
# nb_cov = 2
# nb_cov_infla = 2
# good_fit = False
# viz = "dims"
# # viz = "samples"
# nb_bootstrap = 2

ENH_CLOSED_KEY = "Enhanced Analytic"
ENH_FREE_KEY = "Enhanced"
STD_CLOSED_KEY = "Standard Analytic"
STD_FREE_KEY = "Standard"
PLN = "Pln"
FAIRPLN = "fair_Pln"
ZIPREG = "ZIP"

LABEL_DICT = {
    ENH_CLOSED_KEY: "Enhanced Analytic",
    ENH_FREE_KEY: "Enhanced",
    STD_CLOSED_KEY: "Standard Analytic",
    STD_FREE_KEY: "Standard",
    PLN: "Pln",
    FAIRPLN: "fair_Pln",
    ZIPREG: "ZIP",
}
KEY_MODELS = [
    ENH_CLOSED_KEY,
    STD_FREE_KEY,
    ENH_FREE_KEY,
    STD_CLOSED_KEY,
    PLN,
    FAIRPLN,
    ZIPREG,
]

ELBO_KEY = "ELBO"
TIME_KEY = "TIME"
NBITER_KEY = "NBITER"
REC_KEY = "Reconstruction_error"
B_KEY = "RMSE_B"
OMEGA_KEY = "RMSE_OMEGA"
SIGMA_KEY = "RMSE_SIGMA"
PI_KEY = "RMSE_PI"
B0_KEY = "RMSE_B0"

CRITERION_KEYS = [
    ELBO_KEY,
    REC_KEY,
    OMEGA_KEY,
    SIGMA_KEY,
    B_KEY,
    PI_KEY,
    B0_KEY,
    TIME_KEY,
    NBITER_KEY,
]


def RMSE(t):
    return torch.sqrt(torch.mean(t**2))


VIZ_LABEL = {"ns": r"$n$", "ps": r"$p$"}

COLORS = {
    ENH_FREE_KEY: "cornflowerblue",
    ENH_CLOSED_KEY: "darkblue",
    STD_FREE_KEY: "lightcoral",
    STD_CLOSED_KEY: "darkred",
    PLN: "black",
    FAIRPLN: "grey",
    ZIPREG: "yellow",
}

if viz == "samples":
    abscisses = ns
else:
    abscisses = ps


def fit_models(dict_models):
    for key, model in dict_models.items():
        if key == ZIPREG:
            model.fit(nb_max_iteration=0)
        else:
            if good_fit is True:
                model.fit(tol=0, nb_max_iteration=1000)
            else:
                model.fit(nb_max_iteration=25)
    return dict_models


def get_dict_models(
    endog, exog, exog_inflation, offsets, inflation_formula, fair_endog
):
    sim_models = {
        ENH_FREE_KEY: ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=False,
            add_const=False,
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
            add_const=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
        STD_FREE_KEY: Brute_ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=False,
            add_const=False,
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
            add_const=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
        ),
        PLN: Pln(endog, exog=exog, offsets=offsets),
        FAIRPLN: Pln(fair_endog, exog=exog, offsets=offsets),
        ZIPREG: ZIPln(
            endog,
            exog=exog,
            offsets=offsets,
            add_const_inflation=False,
            add_const=False,
            exog_inflation=exog_inflation,
            zero_inflation_formula=inflation_formula,
            use_closed_form_prob=False,
        ),
    }
    return sim_models


class one_plot:
    def __init__(self, ns_or_ps, viz, inflation_formula, nb_bootstrap):
        self.nb_bootstrap = nb_bootstrap
        self.inflation_formula = inflation_formula
        self.ns_or_ps = ns_or_ps
        self.viz = viz
        if self.viz == "samples":
            self.ns = ns_or_ps
            self.dim = p
        else:
            self.ps = ns_or_ps
            self.n_samples = n
        self.model_criterions = {
            key_model: {
                xscale: {
                    REC_KEY: [],
                    OMEGA_KEY: [],
                    SIGMA_KEY: [],
                    B_KEY: [],
                    PI_KEY: [],
                    ELBO_KEY: [],
                    B0_KEY: [],
                    TIME_KEY: [],
                    NBITER_KEY: [],
                }
                for xscale in self.ns_or_ps
            }
            for key_model in KEY_MODELS
        }

    def build_csv(self):
        for to_viz in self.ns_or_ps:
            if exists(self.abs_name_file):
                print("Getting back data")
                with open(self.abs_name_file, "rb") as fp:
                    self.model_criterions = pickle.load(fp)
            else:
                plnparam = self.get_param(to_viz)
                plnparam._set_mean_proba(mean_infla)
                plnparam._set_gaussian_mean(mean_poiss)
                Sigma = plnparam.covariance
                B = plnparam.coef
                B0 = plnparam.coef_inflation
                print("mean gaussian", torch.mean(plnparam.gaussian_mean))
                print("mean proba", torch.mean(plnparam.proba_inflation))
                for i in tqdm(range(self.nb_bootstrap)):
                    endog, fair_endog = sample_zipln(plnparam, seed=i, return_pln=True)
                    dict_models = get_dict_models(
                        endog,
                        exog=plnparam.exog,
                        offsets=plnparam.offsets,
                        exog_inflation=plnparam.exog_inflation,
                        inflation_formula=self.inflation_formula,
                        fair_endog=fair_endog,
                    )
                    samples_only_zeros = torch.sum(endog, axis=1) == 0
                    dim_only_zeros = torch.sum(endog, axis=0) == 0
                    for model in dict_models.values():
                        model.samples_only_zeros = samples_only_zeros
                        model.dim_only_zeros = dim_only_zeros
                        fit_models(dict_models)
                    self.stock_results(dict_models, to_viz, Sigma, B, B0)
        self.save_criterions()
        data = self.data
        data.to_csv(
            f"csv_computation/{self.viz}_{self.inflation_formula}_not_n_or_p_{self.not_n_or_p}.csv"
        )

    @property
    def doss_viz(self):
        return "samples_viz" if self.viz == "samples" else "dims_viz"

    @property
    def data(self):
        columns = ["model_name", "xscale"]
        columns += CRITERION_KEYS
        data = pd.DataFrame(columns=columns)
        for model_key in KEY_MODELS:
            for xscale in self.ns_or_ps:
                for i in range(self.nb_bootstrap):
                    values = {
                        "model_name": [LABEL_DICT[model_key]],
                        "xscale": [xscale],
                    }
                    for crit_key in CRITERION_KEYS:
                        value = self.model_criterions[model_key][xscale][crit_key][i]
                        if crit_key == ELBO_KEY:
                            value = -value
                        if isinstance(value, torch.Tensor):
                            values[crit_key] = [value.item()]
                        else:
                            values[crit_key] = [value]
                    new_line = pd.DataFrame(values)
                    data = pd.concat((data, new_line))
        return data

    def stock_results(self, dict_models, xscale, Sigma, B, B0):
        for key_model in KEY_MODELS:
            model_fitted = dict_models[key_model]
            lines = ~model_fitted.samples_only_zeros
            cols = ~model_fitted.dim_only_zeros
            Sigma_fair = Sigma
            omega_fair = torch.inverse(Sigma_fair)
            beta_fair = B
            Sigma = Sigma[cols, :][:, cols]
            omega = torch.inverse(Sigma)
            beta = B[:, cols]
            results_model = self.model_criterions[key_model][xscale]
            if key_model != FAIRPLN or key_model != ZIPREG:
                if model_fitted._NAME != "Pln":
                    if model_fitted._zero_inflation_formula == "row-wise":
                        beta_0 = B0[lines, :]
                    elif model_fitted._zero_inflation_formula == "column-wise":
                        beta_0 = B0[:, cols]
                    else:
                        beta_0 = B0
                    results_model[B0_KEY].append(
                        RMSE(model_fitted.coef_inflation - beta_0.cpu())
                    )
                    results_model[PI_KEY].append(
                        RMSE(
                            torch.sigmoid(model_fitted.coef_inflation)
                            - torch.sigmoid(beta_0).cpu()
                        )
                    )
                else:
                    results_model[B0_KEY].append(666)
                    results_model[PI_KEY].append(666)
                rmse_omega = RMSE(torch.inverse(model_fitted.covariance) - omega.cpu())
                results_model[OMEGA_KEY].append(rmse_omega)
                rmse_sigma = RMSE(model_fitted.covariance - Sigma.cpu())
                results_model[SIGMA_KEY].append(rmse_sigma)

                results_model[B_KEY].append(RMSE(model_fitted.coef - beta.cpu()))
            else:
                results_model[B0_KEY].append(666)
                results_model[PI_KEY].append(666)
                if key_model == FAIRPLN:
                    results_model[B_KEY].append(
                        RMSE(model_fitted.coef - beta_fair.cpu())
                    )
                    results_model[OMEGA_KEY].append(
                        RMSE(torch.inverse(model_fitted.covariance) - omega_fair)
                    )
                    results_model[SIGMA_KEY].append(
                        RMSE(model_fitted.covariance - Sigma_fair)
                    )
                else:
                    results_model[B_KEY].append(666)
                    results_model[OMEGA_KEY].append(666)
                    results_model[SIGMA_KEY].append(666)

            if key_model != ZIPREG:
                results_model[REC_KEY].append(model_fitted.reconstruction_error)
                results_model[ELBO_KEY].append(model_fitted.elbo)
                results_model[TIME_KEY].append(
                    model_fitted._criterion_args.running_times[-1]
                )
                results_model[NBITER_KEY].append(model_fitted.nb_iteration_done)
            else:
                results_model[REC_KEY].append(model_fitted.rec_error_init)
                results_model[ELBO_KEY].append(666)
                results_model[TIME_KEY].append(666)
                results_model[NBITER_KEY].append(666)

    def save_criterions(self):
        with open(self.abs_name_file, "wb") as fp:
            pickle.dump(self.model_criterions, fp)

    def get_param(self, to_viz):
        if self.viz == "samples":
            _n = int(to_viz)
            _dim = self.dim
        else:
            _n = self.n_samples
            _dim = int(to_viz)
        if self.inflation_formula == "global":
            _nb_cov_infla = 0
            _add_const_inflation = False
        else:
            _nb_cov_infla = nb_cov_infla
            _add_const_inflation = True
        param = get_simulation_parameters(
            nb_cov=nb_cov,
            add_const=True,
            nb_cov_inflation=_nb_cov_infla,
            zero_inflation_formula=self.inflation_formula,
            n_samples=_n,
            add_const_inflation=_add_const_inflation,
            dim=_dim,
        )
        param._offsets *= 0
        return param

    @property
    def abs_name_file(self):
        return f"results_computation/{self.name_file}"

    @property
    def name_file(self):
        return f"n_or_p_{self.viz}_other_{self.not_n_or_p}_inflation_{self.inflation_formula}"

    @property
    def not_n_or_p(self):
        if self.viz == "samples":
            return self.dim
        else:
            return self.n_samples


for formula in tqdm(["column-wise", "row-wise", "global"]):
    op = one_plot(abscisses, viz, formula, nb_bootstrap)
    op.build_csv()
