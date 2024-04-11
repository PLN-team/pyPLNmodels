import pandas as pd
import numpy as np
import torch

from pyPLNmodels import Pln
from pyPLNmodels.models import Pln_0
from pyPLNmodels.elbos import elbo_pln, elbo_pln_0, r_elbo_pln_0


## microcosm data
data = {}
data["endog"] = pd.read_csv("reduced/endog.csv").drop(columns="Unnamed: 0")
p = data["endog"].shape[1]
data["site"] = pd.read_csv("reduced/site.csv").drop(columns="Unnamed: 0").squeeze()
data["time"] = pd.read_csv("reduced/time.csv").drop(columns="Unnamed: 0").squeeze()
data["site_time"] = data["time"] + data["site"]
best = (data["endog"] > 0).mean(axis=0) > 0.30
data["endog"] = data["endog"].loc[:, best]
data["offsets"] = (
    np.log(
        pd.read_csv("reduced/offsets.csv")
        .drop(columns="Unnamed: 0")
        .values.repeat(p, axis=1)
    )[:, best]
    * 0
)


pln = Pln.from_formula("endog ~ 1", data=data)
n = pln.n_samples
pln.fit(verbose=True)
print(pln)

X = pln.exog

M = torch.from_numpy(
    pd.read_csv("no_covariates_M_20779.csv").drop(columns="Unnamed: 0").values
)
S = torch.from_numpy(
    pd.read_csv("no_covariates_S_20779.csv").drop(columns="Unnamed: 0").values
)
B = torch.from_numpy(
    pd.read_csv("no_covariates_beta_20779.csv").drop(columns="Unnamed: 0").values
)
covariance = torch.from_numpy(
    pd.read_csv("no_covariates_cov_20779.csv").drop(columns="Unnamed: 0").values
)


print(
    "my own loglike",
    n
    * r_elbo_pln_0(
        pln.endog,
        pln.exog,
        pln.offsets,
        pln.latent_mean - pln.exog @ pln.coef,
        pln.latent_sqrt_var,
        pln.covariance,
        pln.coef,
    ),
)
# print('my own loglike centered in 0', n*r_elbo_pln_0(pln.endog, pln.exog, pln.offsets, pln.latent_mean - pln.exog$pln.coef, pln.latent_sqrt_var, pln.covariance, pln.coef))
print(
    "r loglike 0",
    n * r_elbo_pln_0(pln.endog, pln.exog, pln.offsets, M, S, covariance, B),
)
# print('r loglike 0 bis', torch.sum(r_elbo_pln_0_bis(pln.endog, pln.exog, pln.offsets, M , S, covariance, B)))


# zi = Brute_ZIPln.from_formula(
#     "endog ~ 1 + site_time", data=data, zero_inflation_formula="global", use_closed_form_prob=False
# )
# zi.fit(verbose = True, nb_max_iteration = 1000, tol = 0)
# zi.show()
# print(zi)
# zi.viz(colors = data["time"])
# zi.viz_position(colors = data["time"])
