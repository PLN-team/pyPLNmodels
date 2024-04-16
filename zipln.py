import pandas as pd
import numpy as np
import torch

from pyPLNmodels import Pln, Brute_ZIPln
from pyPLNmodels.models import Pln_0
from pyPLNmodels.elbos import elbo_brute_zipln_covariance

B = torch.from_numpy(
    pd.read_csv("zi_no_covariates_beta_106836.csv").drop(columns="Unnamed: 0").values
)
M = torch.from_numpy(
    pd.read_csv("zi_no_covariates_M_106836.csv").drop(columns="Unnamed: 0").values
)
S = torch.from_numpy(
    pd.read_csv("zi_no_covariates_S_106836.csv").drop(columns="Unnamed: 0").values
)
R = torch.from_numpy(
    pd.read_csv("zi_no_covariates_R_106836.csv").drop(columns="Unnamed: 0").values
)
Pi = torch.from_numpy(
    pd.read_csv("zi_no_covariates_Pi_106836.csv").drop(columns="Unnamed: 0").values
)
covariance = torch.from_numpy(
    pd.read_csv("zi_no_covariates_cov_106836.csv").drop(columns="Unnamed: 0").values
)

## microcosm data
data = {}
data["endog"] = pd.read_csv("reduced/endog.csv").drop(columns="Unnamed: 0")
p = data["endog"].shape[1]
data["site"] = pd.read_csv("reduced/site.csv").drop(columns="Unnamed: 0").squeeze()
data["time"] = pd.read_csv("reduced/time.csv").drop(columns="Unnamed: 0").squeeze()
data["site_time"] = data["time"] + data["site"]
best = (data["endog"] > 0).mean(axis=0) > 0.05
data["endog"] = data["endog"].loc[:, best]
data["offsets"] = (
    np.log(
        pd.read_csv("reduced/offsets.csv")
        .drop(columns="Unnamed: 0")
        .values.repeat(p, axis=1)
    )[:, best]
    * 0
)


zi = Brute_ZIPln.from_formula(
    "endog ~ 1",
    data=data,
    zero_inflation_formula="column-wise",
    use_closed_form_prob=False,
)
n = zi.n_samples
zi.fit()
print(zi)
# zi.viz(colors = data["site_time"] )
# zi.viz_positions(colors = data["site_time"])

my_elbo = elbo_brute_zipln_covariance(
    zi.endog,
    zi.exog,
    zi.offsets,
    zi.latent_mean,
    zi.latent_sqrt_var,
    zi.latent_prob,
    zi.covariance,
    zi.coef,
    zi.xinflacoefinfla,
    zi._dirac,
)
print("my elbo", n * my_elbo)

r_elbo = elbo_brute_zipln_covariance(
    zi.endog,
    zi.exog,
    zi.offsets,
    M,
    S,
    (R * zi._dirac).float(),
    covariance,
    B,
    torch.logit(Pi),
    zi._dirac,
)


# print('other', n*r_elbo )
