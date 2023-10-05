from pyPLNmodels.models import PlnPCA, PlnPCAcollection, Pln
from pyPLNmodels import get_real_count_data, get_simulated_count_data
import matplotlib.pyplot as plt
import os
from os.path import exists
import pandas as pd
import numpy as np
import scanpy
from sklearn.preprocessing import LabelEncoder
import pickle
from pyPLNmodels._utils import lissage


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
        axes[2].plot(
            absc, -np.array(model._elbos_list), color=colors[i], linestyle=linestyle
        )
        nb_diff = 3

        tol_list.append(model._plotargs.criterions[-1])

        def plot_normalized_diff(absc, to_plot_orig, ax):
            print("to plot orig", to_plot_orig)
            absc, to_plot_orig = lissage(absc, to_plot_orig, 5)
            print("to plot orig after first lissage", to_plot_orig)
            nb_crit = len(to_plot_orig)
            crit_shifted = to_plot_orig[: nb_crit - nb_diff]
            crit = to_plot_orig[nb_diff:]
            crit = np.abs(
                (np.array(crit) - np.array(crit_shifted))
                / (np.array(crit) + np.array(crit_shifted))
            )

            absc_crit = absc[nb_diff:]
            absc_crit, crit = lissage(absc_crit, crit, 5)
            crit = np.array(crit)
            print('crit :', crit)
            ax.plot(absc_crit, crit, color=colors[i], linestyle=linestyle)

        plot_normalized_diff(absc, model._elbos_list, axes[5])


class mimick_col:
    def __init__(self, col):
        self.col = {}
        self.ranks = col.ranks
        for rank in col.ranks:
            self.col[rank] = mimick_pca(col[rank])

    def values(self):
        return self.col.values()


class mimick_pca:
    def __init__(self, pca):
        self._plotargs = pca._plotargs
        self.mse_beta_list = pca.mse_beta_list
        self.rank = pca.rank
        self.mse_Sigma_list = pca.mse_Sigma_list
        self.norm_list_beta = pca.norm_list_beta
        self.norm_list_Sigma = pca.norm_list_Sigma
        self._elbos_list = pca._elbos_list
        self.scores_predictor = pca.scores_predictor
        self.mse_latent_mean = pca.mse_latent_mean
        self.mse_latent_sqrt_var = pca.mse_latent_sqrt_var
        self.mse_predictions = pca.mse_predictions
        self.mse_latent_variables = pca.mse_latent_variables


def mimick_col_from_file(name_file):
    with open(name_file, "rb") as fp:
        col = pickle.load(fp)
    return col


def save_col(col, name):
    saveable_col = mimick_col(col)
    with open(name, "wb") as fp:
        pickle.dump(saveable_col, fp)


def plot_n_or_dim(n, dim, nb_fitting, nb_max_iter, batch_size):
    counts, GT, _ = get_sc_mark_data(dim=dim, max_n=n)
    print("Y shape", counts.shape)
    covariates = None
    offsets = None
    # counts, covariates, offsets = get_simulated_count_data(seed = 0)
    ranks = [4, 12, 15]  # , 40, 80, 120, 180, 250, 500]
    name_no_batch = f"results/no_batch_ranks_{ranks}_n_{n}_dim_{dim}_nbfitting_{nb_fitting}_nbiter_{nb_max_iter}"
    name_batch = f"results/batch_{batch_size}ranks_{ranks}_n_{n}_dim_{dim}_nbfitting_{nb_fitting}_nbiter_{nb_max_iter}"
    if exists(name_batch) is False:
        pln = Pln(counts, GT=GT)
        pln.fit(verbose=True, nb_max_iteration=nb_max_iter, tol=1e-4)
        true_beta = pln.coef
        true_Sigma = pln.covariance
    if exists(name_no_batch) is False:
        col = PlnPCAcollection(
            counts,
            exog=covariates,
            offsets=offsets,
            ranks=ranks,
            true_Sigma=true_Sigma,
            true_beta=true_beta,
            GT=GT,
        )
        col.fit(
            tol=1e-7, nb_max_iteration=nb_max_iter, verbose=True, nb_fitting=nb_fitting
        )
        save_col(col, name_no_batch)
    else:
        col = mimick_col_from_file(name_no_batch)
    if exists(name_batch) is False:
        col_batch = PlnPCAcollection(
            counts,
            exog=covariates,
            offsets=offsets,
            ranks=ranks,
            true_Sigma=true_Sigma,
            true_beta=true_beta,
            GT=GT,
        )
        col_batch.fit(
            tol=0,
            nb_max_iteration=nb_max_iter,
            verbose=True,
            batch_size=batch_size,
            nb_fitting=nb_fitting,
        )
        save_col(col_batch, name_batch)
    else:
        col_batch = mimick_col_from_file(name_batch)
    nb_cols = 3
    fig, mp_axes = plt.subplots(2, nb_cols, figsize=(20, 20))
    colors = np.linspace(0, 200, len(col.ranks))
    axes = {}
    for i in range(nb_cols):
        axes[i] = mp_axes[0, i]
        axes[nb_cols + i] = mp_axes[1, i]
    colors /= 235
    colors = colors.reshape(-1, 1)
    colors = np.repeat(colors, 3, axis=1)
    tols = []
    tols_batch = []
    # plot_collection(col, axes, colors, tols, linestyle="-")
    plot_collection(col_batch, axes, colors, tols_batch, linestyle="--")
    # axes[5].plot(ranks, tols, label="Criterion at last iteration", color="black")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[4].legend()

    axes[0].set_yscale("log")
    axes[0].set_title(r"$\hat \beta - \beta^{\star}$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\hat \Sigma - \Sigma^{\star}$")
    axes[2].set_yscale("log")
    axes[2].set_title(r"log(ELBO)")
    axes[3].plot(
        ranks,
        tols_batch,
        label="Criterion at last iteration",
        linestyle="--",
        color="black",
    )
    axes[3].set_title(r"tolerance at last iteration")
    axes[3].set_yscale("log")
    axes[4].set_title("Scores predictor")
    axes[5].set_title("normalized diff elbo")

    plt.savefig(f"n_{n}_p_{dim}_ranks_{ranks}.pdf", format="pdf")
    plt.show()


plot_n_or_dim(450, 100, 100, nb_max_iter=400000, batch_size=20)
