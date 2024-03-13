from pyPLNmodels import Pln, ZIPln, get_simulation_parameters, sample_zipln
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


n_samples = 1000
parameters = get_simulation_parameters(
    zero_inflation_formula="global", n_samples=n_samples, dim=200
)

parameters._set_gaussian_mean(2)
parameters._set_mean_proba(0.3)

exog = parameters.exog
exog_inflation = parameters.exog_inflation
offsets = parameters.offsets
true_Sigma = parameters.covariance
true_Omega = torch.inverse(true_Sigma)
true_coef = parameters.coef

nb_iter = 400
endog_zi, endog_pln = sample_zipln(parameters, return_pln=True)


print(
    "percentage of zeros for the zero inflated data",
    torch.mean((endog_zi == 0).float()),
)
print(
    "percentage of zeros for the non inflated data",
    torch.mean((endog_pln == 0).float()),
)

zi = ZIPln(endog_zi, exog=exog, zero_inflation_formula="global", offsets=offsets)
zi.fit(tol=0, nb_max_iteration=nb_iter)

fairpln = Pln(endog_pln, exog=exog, offsets=offsets)
fairpln.fit(tol=0, nb_max_iteration=nb_iter)

pln = Pln(endog_zi, exog=exog, offsets=offsets)
pln.fit(tol=0, nb_max_iteration=nb_iter)

fig, axes = plt.subplots(2, 2, figsize=(10, 20))


def rmse(t):
    return np.round(torch.sqrt(torch.mean(t**2)), 4)


def check_covariance_coef(model, ax, label):
    Sigma = model.covariance
    Omega = torch.inverse(Sigma)
    coef = model.coef
    mse_Sigma = rmse(Sigma - true_Sigma)
    mse_Omega = rmse(Omega - true_Omega)
    mse_coef = rmse(coef - true_coef)
    sns.heatmap(Omega, ax=ax)
    title = rf"{label}__MSE($\Sigma$){mse_Sigma}, MSE($\Omega$){mse_Omega}, MSE($B$){mse_coef}"
    ax.set_title(title)


sns.heatmap(true_Omega, ax=axes[0, 0])
axes[0, 0].set_title("True Omega")
check_covariance_coef(zi, axes[1, 0], label="ZIPln on ZIPln data")
check_covariance_coef(fairpln, axes[0, 1], label="Pln on pln data.")
check_covariance_coef(pln, axes[1, 1], label="Pln on ZIpln data.")

fig.suptitle("Comparison of estimation of Omegas")
plt.show()


# endog, exog, offsets = get_zipln_simulated_count_data()
