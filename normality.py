import torch
from pyPLNmodels import sample_pln, Pln, get_simulation_parameters
import math
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
from sandwich import Fisher_Pln, normalizing, compute_rmse

# nb_max_iter = 800
# nb_seed = 20
# ns = [250, 500, 1000, 2000]
# dims = [10, 20, 40]
# nb_covs = [1, 2, 3, 4]
# mean_gaussian = 2


def rmse(t):
    return torch.sqrt(torch.mean(t**2))


nb_max_iter = 800
nb_seed_param = 1
nb_seed_count = 100
ns = [500, 1000, 2000][::-1]
dims = [25, 50, 100][::-1]
nb_covs = [1, 2, 3, 4][::-1]
mean_gaussian = 2

N_SAMPLES_KEY = "n_samples"
P_KEY = "p"
DIM_NUMBER_KEY = "dim_number"
NB_COV_KEY = "nb_cov"
SEED_PARAM_KEY = "seed_param"
SEED_COUNT_KEY = "seed_count"
FISHER_KEY = "Variational Fisher Information"
SANDWICH_KEY = "Sandwich based Information"
RMSE_SIGMA_KEY = "RMSE_Sigma"
RMSE_B_KEY = "RMSE_B"


LIST_KEY = [
    N_SAMPLES_KEY,
    P_KEY,
    DIM_NUMBER_KEY,
    NB_COV_KEY,
    SEED_PARAM_KEY,
    SEED_COUNT_KEY,
    FISHER_KEY,
    SANDWICH_KEY,
    RMSE_B_KEY,
    RMSE_SIGMA_KEY,
]


def get_each_gaussian(ns, nb_cov, dim):
    tmp_df = pd.DataFrame(columns=LIST_KEY)
    for seed_param in range(nb_seed_param):
        sim_param = get_simulation_parameters(
            n_samples=max(ns),
            dim=dim,
            seed=seed_param,
            nb_cov=nb_cov - 1,
            add_const=True,
            mean_gaussian=mean_gaussian,
        )
        sim_param._exog = torch.from_numpy(
            np.random.multinomial(1, [1 / nb_cov] * nb_cov, size=sim_param.n_samples)
        ).double()
        for seed_count in range(nb_seed_count):
            _endog = sample_pln(sim_param, seed=seed_count)
            _exog = sim_param.exog
            _offsets = sim_param.offsets
            true_covariance = sim_param.covariance
            true_coef = sim_param.coef
            XB = _exog @ true_coef
            print(" mean XB", torch.mean(XB))
            print("min XB", torch.min(XB))
            for i, n in enumerate(ns):
                dict_df = {l: [] for l in LIST_KEY}

                endog = _endog[:n]
                exog = _exog[:n]
                print("endog shape", endog.shape)
                pln = Pln(endog, exog=exog, add_const=False)
                pln.fit(nb_max_epoch=nb_max_iter, tol=1e-8)

                A = torch.exp(
                    pln.offsets + pln.latent_mean + 0.5 * pln.latent_sqrt_var**2
                )

                test = Fisher_Pln(
                    pln.endog,
                    A,
                    pln.exog,
                    pln.nb_cov,
                    pln.dim,
                    pln.n_samples,
                    pln.latent_sqrt_var,
                    torch.inverse(pln.covariance),
                    pln.covariance,
                )

                var_sandwich = test.getInvSandwich()
                var_variational = test.getInvFisher()

                N01_sandwich = normalizing(
                    pln.coef, true_coef, var_sandwich, pln.n_samples
                )
                N01_fisher = normalizing(
                    pln.coef, true_coef, var_variational, pln.n_samples
                )
                dict_df[N_SAMPLES_KEY] = [n] * len(N01_sandwich)
                dict_df[P_KEY] = [dim] * len(N01_sandwich)
                dict_df[DIM_NUMBER_KEY] = np.arange(len(N01_sandwich)).tolist()
                dict_df[NB_COV_KEY] = [nb_cov] * len(N01_sandwich)
                dict_df[SEED_PARAM_KEY] = [seed_param] * len(N01_sandwich)
                dict_df[SEED_COUNT_KEY] = [seed_count] * len(N01_sandwich)
                dict_df[FISHER_KEY] = N01_fisher.tolist()
                dict_df[SANDWICH_KEY] = N01_sandwich.tolist()
                dict_df[RMSE_B_KEY] = [rmse(pln.coef - true_coef).item()] * len(
                    N01_sandwich
                )
                dict_df[RMSE_SIGMA_KEY] = [
                    rmse(pln.covariance - true_covariance).item()
                ] * len(N01_sandwich)

                _df = pd.DataFrame.from_dict(dict_df)
                tmp_df = pd.concat([tmp_df, _df], axis=0)
    return tmp_df


df_n01 = pd.DataFrame(columns=LIST_KEY)

for dim in tqdm(dims):
    print("dim:", dim)
    for nb_cov in nb_covs:
        print("nb_cov", nb_cov)
        tmp_df = get_each_gaussian(ns, nb_cov, dim)
        df_n01 = pd.concat([df_n01, tmp_df], axis=0)

df_n01.to_csv(
    f"csvs/res_nb_seed_count{nb_seed_count}_nb_seed_param_{nb_seed_param}.csv"
)
