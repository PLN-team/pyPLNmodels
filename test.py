from pyPLNmodels import (
    PlnPCA,
    Pln,
    PlnPCAcollection,
    get_simulated_count_data,
    get_real_count_data,
    get_simulation_parameters,
    sample,
)
from pyPLNmodels.models import BIG
from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels.elbos import _elbo_big
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import statsmodels as sm
import pandas as pd
from sklearn.decomposition import PCA

t = get_simulation_parameters()
t.covariates = None
t.offsets *= 0
counts = sample(t, distrib="BIG")
covariates = t.covariates
n, p = counts.shape
latent_mean = torch.zeros(n, p).requires_grad_(True)
latent_sqrt_var = torch.ones(n, p).requires_grad_(True)
coef = _closed_formula_coef(covariates, latent_mean)
covariance = _closed_formula_covariance(
    covariates, latent_mean, latent_sqrt_var, coef, n
)
ksi = torch.ones(n, p).requires_grad_(True)

optim = torch.optim.Rprop([latent_mean, latent_sqrt_var, ksi], lr=0.001)

nb_iter = 400
for i in range(nb_iter):
    loss = -_elbo_big(
        counts, covariates, latent_mean, latent_sqrt_var, covariance, coef, ksi
    )
    loss.backward()
    optim.step()
    optim.zero_grad()
    coef = _closed_formula_coef(covariates, latent_mean)
    covariance = _closed_formula_covariance(
        covariates, latent_mean, latent_sqrt_var, coef, n
    )

sns.heatmap(covariance.detach())
plt.show()


# counts = sample(t, distrib = "PLN")
# big = BIG(counts, covariates = t.covariates, offsets = t.offsets)
# big.fit(nb_max_iteration = 1500, tol = 0)
# big.show()
# sns.heatmap(t.covariance)
# plt.show()
