from pyPLNmodels.models import PlnPCA, PlnPCAcollection, Pln
from pyPLNmodels import get_real_count_data, get_simulated_count_data
import matplotlib.pyplot as plt
import os
from os.path import exists
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


def plot_collection(col, axes, colors, tol_list, linestyle):
    for i, model in enumerate(col.values()):
        absc = model._plotargs.running_times
        if linestyle == "--":
            label = f"rank {model.rank} mini-batch"
        else:
            label = f"rank {model.rank} full-batch"
        axes[0].plot(
            absc,
            model.mse_beta_list,
            label=label,
            color=colors[i],
            linestyle=linestyle,
        )
        axes[1].plot(absc, model.mse_Sigma_list, color=colors[i], linestyle=linestyle)
        axes[2].plot(absc, model.norm_list_beta, color=colors[i], linestyle=linestyle)
        axes[3].plot(
            absc, -np.array(model._elbos_list), color=colors[i], linestyle=linestyle
        )
        axes[4].plot(absc, model.norm_list_Sigma, color=colors[i], linestyle=linestyle)
        axes[6].plot(absc, model.scores_predictor, color=colors[i], linestyle=linestyle)
        axes[7].plot(
            absc, model._plotargs.criterions, color=colors[i], linestyle=linestyle
        )
        tol_list.append(model._plotargs.criterions[-1])


class mimick_pca:
    def __init__(self, pca):
        self._plotargs = pca._plotargs
        self.mse_beta_list = pca.mse_beta_list
        self.rank = pca.rank
        self.mse_Sigma_list = pca.mse_Sigma_list
        self.norm_list_beta = pca.norm_list_beta
        self.norm_list_Sigma = pca.norm_list_Sigma
        self._elbos_list = pca._elbos_list
        self.scores_xgboost = pca.scores_xgboost


def save_pca(pca):
    col = {rank: {} for rank in pca.ranks}

    for rank in pca.ranks:
        current_model = col[rank]
        current_model
        col[rank]


def plot_n_or_dim(n, dim, max_rt):
    counts, GT, _ = get_sc_mark_data(dim=dim, max_n=n)
    print("Y shape", counts.shape)
    covariates = None
    offsets = None
    pln = Pln(counts, exog=covariates, offsets=offsets, GT=GT)
    pln.fit(verbose=True)
    true_beta = pln.coef
    true_Sigma = pln.covariance
    # counts, covariates, offsets = get_simulated_count_data(seed = 0)
    ranks = [3, 4, 12, 30]  # , 40, 80, 120, 180, 250, 500]
    name_no_batch = f"results/no_batch_ranks_{ranks}_n_{n}_dim_{dim}_maxrt_{max_rt}"
    name_batch = f"results/batch_ranks_{ranks}_n_{n}_dim_{dim}_maxrt_{max_rt}"
    nb_max_iter = 30000
    if exists(name_no_batch) is False:
        pca = PlnPCAcollection(
            counts,
            exog=covariates,
            offsets=offsets,
            ranks=ranks,
            true_Sigma=true_Sigma,
            true_beta=true_beta,
            GT=GT,
        )
        pca.fit(tol=0, nb_max_iteration=nb_max_iter, verbose=True, max_rt=max_rt)
        save_pca(pca)
    else:
        pca = mimick_col(name_no_batch)
    if exists(name_batch) is False:
        pca_batch = PlnPCAcollection(
            counts,
            exog=covariates,
            offsets=offsets,
            ranks=ranks,
            true_Sigma=true_Sigma,
            true_beta=true_beta,
            GT=GT,
        )
        pca_batch.fit(
            tol=0,
            nb_max_iteration=nb_max_iter,
            verbose=True,
            batch_size=300,
            max_rt=max_rt,
        )
        save_pca(pca_batch)
    else:
        pca_batch = mimick_col(name_batch)
    fig, mp_axes = plt.subplots(2, 4, figsize=(20, 20))
    colors = np.linspace(0, 200, len(pca.ranks))
    axes = {}
    for i in range(4):
        axes[i] = mp_axes[0, i]
        axes[4 + i] = mp_axes[1, i]
    colors /= 235
    colors = colors.reshape(-1, 1)
    colors = np.repeat(colors, 3, axis=1)
    tols = []
    tols_batch = []
    plot_collection(pca, axes, colors, tols, linestyle="-")
    plot_collection(pca_batch, axes, colors, tols_batch, linestyle="--")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[4].legend()
    axes[5].legend()
    axes[0].set_yscale("log")
    axes[0].set_title(r"$\hat \beta - \beta^{\star}$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\hat \Sigma - \Sigma^{\star}$")
    axes[2].set_yscale("log")
    axes[2].set_title(r"$\|\beta\|$")
    axes[3].set_yscale("log")
    axes[3].set_title(r"log(ELBO)")
    axes[4].set_yscale("log")
    axes[4].set_title(r"$\|\Sigma\|$")
    axes[5].plot(ranks, tols, label="Criterion at last iteration", color="black")
    axes[5].plot(
        ranks,
        tols_batch,
        label="Criterion at last iteration",
        linestyle="--",
        color="black",
    )
    axes[5].set_title(r"tolerance at last iteration")
    axes[5].set_yscale("log")
    axes[6].set_title("Scores predictor")
    axes[7].set_yscale("log")
    axes[7].set_title("tolerance")
    plt.savefig(f"n_{n}_p_{dim}_ranks_{ranks}.pdf", format="pdf")
    plt.show()


plot_n_or_dim(1000, 150, 10)
