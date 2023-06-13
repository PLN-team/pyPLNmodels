import torch  # pylint:disable=[C0114]

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
from pyPLNmodels.elbos import _elbo_big, profiled_elbo_big
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import statsmodels as sm
import pandas as pd
from sklearn.decomposition import PCA

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

t = get_simulation_parameters(n_samples=200)
t.offsets *= 0
counts = sample(t, distrib="BIG")
n, p = counts.shape
t.offsets *= 0
covariates = t.covariates
latent_mean = torch.zeros(n, p, device=DEVICE).requires_grad_(True)
latent_sqrt_var = torch.ones(n, p, device=DEVICE).requires_grad_(True)
coef = _closed_formula_coef(covariates, latent_mean)
covariance = _closed_formula_covariance(
    covariates, latent_mean, latent_sqrt_var, coef, n
)
ksi = torch.ones(n, p, device=DEVICE).requires_grad_(True)

optim = torch.optim.Adam([latent_mean, latent_sqrt_var, ksi], lr=0.01)


pln = Pln(counts, covariates=covariates, offsets=t.offsets, add_const=False)
pln.fit()
covariance = pln.covariance
sns.heatmap(covariance)
plt.show()
nb_iter = 1000
elbo = np.zeros([nb_iter])
for i in range(nb_iter):
    loss = -profiled_elbo_big(counts, covariates, latent_mean, latent_sqrt_var, ksi)
    loss.backward()
    optim.step()
    optim.zero_grad()
    elbo[i] = loss.item()

plt.plot(elbo)
plt.yscale("log", base=10)
plt.show()

coef = _closed_formula_coef(covariates, latent_mean)
covariance = _closed_formula_covariance(
    covariates, latent_mean, latent_sqrt_var, coef, n
)
sns.heatmap(covariance.detach())
plt.show()
sns.heatmap(t.covariance)
plt.show()
print("mse:", torch.mean((t.covariance - pln.covariance) ** 2))
