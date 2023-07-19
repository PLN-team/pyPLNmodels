from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scanpy
from sklearn.preprocessing import LabelEncoder


def get_sc_mark_data(max_class=28, max_n=200, dim=100):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:max_n]
    GT_name = data.obs["standard_true_celltype_v5"][:max_n]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    unique, index = np.unique(GT, return_counts=True)
    enough_elem = index > 15
    classes_with_enough_elem = unique[enough_elem]
    filter_bis = np.isin(GT, classes_with_enough_elem)
    mask = filter * filter_bis
    GT = GT[mask]
    GT_name = GT_name[mask]
    Y = Y[mask]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    var = np.var(Y, axis=0)
    most_variables = np.argsort(var)[-dim:]
    Y = Y[:, most_variables]
    return Y, GT, list(GT_name.values.__array__())


def plot_n_or_dim(n, dim):
    counts, _, _ = get_sc_mark_data(dim=dim, max_n=n)
    print("Y shape", counts.shape)
    covariates = None
    offsets = None
    pln = PLN(counts, covariates, offsets)
    pln.fit(verbose=True)
    true_beta = pln.coef
    true_Sigma = pln.covariance
    # counts, covariates, offsets = get_simulated_count_data(seed = 0)
    ranks = [3, 4, 5, 7, 9, 12, 15, 20, 30, 40]
    pca = PLNPCA(
        counts,
        covariates,
        offsets,
        ranks=ranks,
        true_Sigma=true_Sigma,
        true_beta=true_beta,
    )
    nb_max_iter = 200
    pca.fit(tol=0, nb_max_iteration=nb_max_iter, verbose=True)
    fig, axes = plt.subplots(5)
    colors = np.linspace(0, 200, len(pca.ranks))
    colors /= 235
    colors = colors.reshape(-1, 1)
    colors = np.repeat(colors, 3, axis=1)
    tols = []
    for i, model in enumerate(pca.models):
        # plt.plot(1/np.array(model.plotargs.criterions[50:]),model.mse_beta_list[50:], label = f"{model.rank} beta", linestyle = '--')
        # plt.plot(1/np.array(model.plotargs.criterions[50:]),model.mse_Sigma_list[50:], label = f"{model.rank}  Sigma")
        # nb_tols = 7
        # nb_iter = 600
        # print("current_tol", model.plotargs.criterions[nb_iter])
        # tols = np.logspace(-7, 1, nb_tols)
        # firsts = [np.argmin(model.plotargs.criterions > tol) for tol in tols]
        # print("criterion:", model.plotargs.criterions)
        # print('first :', firsts)
        absc = np.arange(len(model.mse_beta_list))
        axes[0].plot(
            absc,
            model.mse_beta_list,
            label=f"{model.rank} beta",
            linestyle="--",
            color=colors[i],
        )
        axes[1].plot(
            absc, model.mse_Sigma_list, label=f"{model.rank}  Sigma", color=colors[i]
        )
        axes[2].plot(
            absc, model.norm_list_C, label=f"{model.rank}  norm C", color=colors[i]
        )
        axes[3].plot(
            absc, model.norm_list_C, label=f"{model.rank}  norm beta", color=colors[i]
        )
        tols.append(model.plotargs.criterions[-1])
        # for i in range(nb_tols):
        # plt.axvline(firsts[i], label = tols[i], linestyle = '--')
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    axes[4].plot(ranks, tols, label="Criterion at last iteration")
    axes[4].set_yscale("log")
    plt.show()


plot_n_or_dim(500, 2000)
