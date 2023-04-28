from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

os.chdir("./pyPLNmodels/")


def plot_n_or_dim(n, dim):
    counts = get_real_count_data(dim=dim, n_samples=n)
    print("Y shape", counts.shape)
    covariates = None
    offsets = None
    pln = PLN(counts, covariates, offsets)
    pln.fit()
    true_beta = pln.coef
    true_Sigma = pln.covariance
    # counts, covariates, offsets = get_simulated_count_data(seed = 0)
    pca = PLNPCA(
        counts,
        covariates,
        offsets,
        ranks=[3, 4, 5, 7, 9, 12, 15, 20, 30, 40],
        true_Sigma=true_Sigma,
        true_beta=true_beta,
    )
    pca.fit(tol=0, nb_max_iteration=700)
    for model in pca.models:
        # plt.plot(1/np.array(model.plotargs.criterions[50:]),model.mse_beta_list[50:], label = f"{model.rank} beta", linestyle = '--')
        # plt.plot(1/np.array(model.plotargs.criterions[50:]),model.mse_Sigma_list[50:], label = f"{model.rank}  Sigma")
        nb_tols = 7
        nb_iter = 600
        print("current_tol", model.plotargs.criterions[nb_iter])
        tols = np.logspace(-7, 1, nb_tols)
        firsts = [np.argmin(model.plotargs.criterions > tol) for tol in tols]
        # print("criterion:", model.plotargs.criterions)
        # print('first :', firsts)
        absc = np.arange(len(model.mse_beta_list))
        plt.plot(absc, model.mse_beta_list, label=f"{model.rank} beta", linestyle="--")
        plt.plot(absc, model.mse_Sigma_list, label=f"{model.rank}  Sigma")
        # for i in range(nb_tols):
        # plt.axvline(firsts[i], label = tols[i], linestyle = '--')
    plt.legend()
    plt.show()


plot_n_or_dim(1000, 40)
# pca.fit()
# print(pca)
# data = pd.DataFrame(counts)
# pln = PLN("counts~1", data)
# pln.fit()
# print(pln)
# pcamodel = pca.best_model()
# pcamodel.save()
# model = PLNPCA([4])[4]

# model.load()
# # pln.fit(counts, covariates, offsets, tol=0.1)
