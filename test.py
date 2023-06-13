import torch  # pylint:disable=[C0114]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
import seaborn as sns

## Extremely favorable settings
param = get_simulation_parameters(n_samples=1000, dim=10, nb_cov=0)
param.offsets *= 0

counts, gaussian, ksi = sample(param, distrib="BIG", return_latent=True)
n, p = counts.shape

covariates = param.covariates
latent_mean = torch.zeros(n, p, device=DEVICE).requires_grad_(True)
latent_sqrt_var = torch.ones(n, p, device=DEVICE).requires_grad_(True)
ksi = torch.ones(n, p, device=DEVICE).requires_grad_(True)

optim = torch.optim.Rprop([latent_mean, latent_sqrt_var, ksi], lr=0.1)

## Bernoulli Logistic Multivariate Normal
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
plt.xscale("log", base=10)
plt.show()


coef = _closed_formula_coef(covariates, latent_mean)
covariance = _closed_formula_covariance(
    covariates, latent_mean, latent_sqrt_var, coef, n
)

fig, axs = plt.subplots(4, 2)
fig.suptitle('Correlation matrices (True, Latent Gaussian layer, Latent Prob., Obser. Counts)')
axs[0,0].hist(param.covariance.cpu().numpy())
axs[0,1].imshow(param.covariance.cpu().numpy())
axs[1,0].hist(np.corrcoef(gaussian.T.cpu()))
axs[1,1].imshow(np.corrcoef(gaussian.T.cpu()))
axs[2,0].hist(np.corrcoef(torch.sigmoid(gaussian).T.cpu()))
axs[2,1].imshow(np.corrcoef(torch.sigmoid(gaussian).T.cpu()))
axs[3,0].hist(np.corrcoef(counts.T.cpu()))
axs[3,1].imshow(np.corrcoef(counts.T.cpu()))
fig.tight_layout()
plt.show()

print("mse:", torch.mean((param.covariance - covariance) ** 2)/torch.mean((param.covariance) ** 2))
print("mse:", torch.mean((param.coef - coef) ** 2)/torch.mean((param.coef) ** 2))

fig, axs = plt.subplots(4, 2)
fig.suptitle('Correlation matrices (True, Latent Gaussian layer, Latent Prob., Obser. Counts)')
axs[0,0].hist(latent_mean.detach().cpu().numpy())
axs[0,1].hist(latent_sqrt_var.detach().cpu().numpy())
axs[1,0].hist(np.corrcoef(gaussian.T.cpu()))
axs[1,1].imshow(np.corrcoef(gaussian.T.cpu()))
axs[2,0].hist(np.corrcoef(torch.sigmoid(gaussian).T.cpu()))
axs[2,1].imshow(np.corrcoef(torch.sigmoid(gaussian).T.cpu()))
axs[3,0].hist(np.corrcoef(counts.T.cpu()))
axs[3,1].imshow(np.corrcoef(counts.T.cpu()))
fig.tight_layout()
plt.show()
