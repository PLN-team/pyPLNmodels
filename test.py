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


def plot_collection(
    col, axes, dict_colors, tol_list, linestyle,dict_ranks, batch_size=None
):
    if batch_size is None:
        color = "red"
    else:
        color = dict_colors[batch_size]

    for i, model in enumerate(col.values()):
        marker = dict_ranks[model.rank]
        if i>-1:
            absc = model._plotargs.running_times
            if linestyle == "--":
                label = f"rank {model.rank} BS={batch_size} "
            else:
                label = f"rank {model.rank} full-batch"
            axes[0].plot(
                absc,
                model.mse_beta_list,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            axes[1].plot(
                absc,
                model.mse_Sigma_list,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            axes[2].plot(
                absc,
                -np.array(model._elbos_list),
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            axes[6].plot(
                absc,
                model._plotargs.new_criterion_list,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )

            # def derive(criter, absc):
            #     dx = np.diff(absc)
            #     df = np.diff(criter) / dx
            #     return df
            # derivative = np.abs(derive(to_derive, absc))
            # nb_lissage = 50
            # absc_der = absc[1:]
            # beta = 0.03
            # moyenne_glissante = 0
            # lissed_derivative = []
            # for el in derivative:
            #     moyenne_glissante = moyenne_glissante*(1-beta) + beta*el
            #     lissed_derivative.append(moyenne_glissante)

            # # absc_der, lissed_derivative = lissage(absc[:-1], derivative, nb_lissage)
            # axes[7].plot(
            #         absc_der,
            #     lissed_derivative,
            #     color=color,
            #     linestyle=linestyle,
            #     marker=marker,
            # )
            # f_second  = np.abs(derive(lissed_derivative, absc_der))
            # # absc_hessian, lissed_hessian = lissage(absc_der[:-1], f_second, nb_lissage)
            # absc_hessian = absc_der[1:]
            # lissed_hessian = []
            # for el in f_second:
            #     moyenne_glissante = moyenne_glissante*(1-beta) + beta*el
            #     lissed_hessian.append(moyenne_glissante)

            # axes[5].plot(
            #         absc_hessian,
            #         lissed_hessian,
            #     color=color,
            #     linestyle=linestyle,
            #     marker=marker,
            # )

            tol_list.append(model._plotargs.new_criterion_list[-1])

        def plot_normalized_diff(absc, to_plot_orig, ax, nb_lissage, nb_diff):
            # print("to plot orig", to_plot_orig)
            absc, to_plot_orig = lissage(absc, to_plot_orig, nb_lissage)
            # print("to plot orig after first lissage", to_plot_orig)
            nb_crit = len(to_plot_orig)
            crit_shifted = to_plot_orig[: nb_crit - nb_diff]
            crit = to_plot_orig[nb_diff:]
            crit = np.abs(
                (np.array(crit) - np.array(crit_shifted))
                / (np.array(crit) + np.array(crit_shifted))
            )

            absc_crit = absc[nb_diff:]
            absc_crit, crit = lissage(absc_crit, crit, nb_lissage)
            crit = np.array(crit)
            ax.plot(absc_crit, crit, color=color, linestyle=linestyle)

        nb_diff = 30
        nb_lissage = 200
        # plot_normalized_diff(absc, -np.cumsum(model._elbos_list), axes[6], 1,1)

        # plot_normalized_diff(absc, model._elbos_list, axes[5], nb_lissage, nb_diff)


class mimick_col:
    def __init__(self, col):
        self.col = {}
        self.ranks = col.ranks
        rank0 = col.ranks[0]
        self.true_Sigma = col[rank0].true_Sigma
        self.true_beta = col[rank0].true_beta
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
        self.true_Sigma = pca.true_Sigma
        self.true_beta = pca.true_beta


def mimick_col_from_file(name_file):
    with open(name_file, "rb") as fp:
        col = pickle.load(fp)
    return col


def save_col(col, name):
    saveable_col = mimick_col(col)
    with open(name, "wb") as fp:
        pickle.dump(saveable_col, fp)


def plot_n_or_dim(n, dim, dict_ranks, nb_max_iter, batches_size):
    ranks = list(dict_ranks.keys())
    counts, GT, _ = get_sc_mark_data(dim=dim, max_n=n)
    print("Y shape", counts.shape)
    covariates = None
    offsets = None
    # counts, covariates, offsets = get_simulated_count_data(seed = 0)
    name_no_batch = (
        f"results/no_batch_ranks_{ranks}_n_{n}_dim_{dim}_nbiter_{nb_max_iter}"
    )
    nb_cols = 4
    fig, mp_axes = plt.subplots(2, nb_cols, figsize=(20, 20))
    colors = np.linspace(0, 200, len(batches_size))
    axes = {}
    for i in range(nb_cols):
        axes[i] = mp_axes[0, i]
        axes[nb_cols + i] = mp_axes[1, i]
    colors /= 235
    colors = colors.reshape(-1, 1)
    colors = np.repeat(colors, 3, axis=1)
    dict_colors = {}
    for batch_size, color in zip(batches_size, colors):
        dict_colors[batch_size] = color

    tols = []
    if exists(name_no_batch) is False:
        pln = Pln(counts, GT=GT)
        pln.fit(verbose=True, nb_max_iteration=nb_max_iter, tol=1e-4)
        true_beta = pln.coef
        true_Sigma = pln.covariance
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
            tol=0.00,
            nb_max_iteration=nb_max_iter,
            verbose=True,
        )
        save_col(col, name_no_batch)
    else:
        col = mimick_col_from_file(name_no_batch)
        true_Sigma = col.true_Sigma
        true_beta = col.true_beta
    plot_collection(col, axes, dict_colors, tols,dict_ranks = dict_ranks, linestyle="-")
    tols_batches = {}
    for batch_size in batches_size:
        tols_batch = []
        name_batch = f"results/batch_{batch_size}ranks_{ranks}_n_{n}_dim_{dim}_nbiter_{nb_max_iter}"
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
                tol=0.00,
                nb_max_iteration=nb_max_iter,
                verbose=True,
                batch_size=batch_size,
            )
            save_col(col_batch, name_batch)
        else:
            col_batch = mimick_col_from_file(name_batch)
        plot_collection(
            col_batch,
            axes,
            dict_colors,
            tols_batch,
            linestyle="--",
            dict_ranks=dict_ranks,
            batch_size=batch_size,
        )
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
    axes[5].set_title("f second cumsum ")
    axes[5].set_yscale("log")
    axes[6].set_title("criterion")
    axes[6].set_yscale("log")
    axes[7].set_title("f' cumsum elbo")
    axes[7].set_yscale("log")

    plt.savefig(f"n_{n}_p_{dim}_ranks_{ranks}.pdf", format="pdf")
    plt.show()


dict_ranks = {4:"v", 12:"o", 15:"<", 25:"1"}  # , 40, 80, 120, 180, 250, 500]
batches_size = [20,50,100,500, 1000]
plot_n_or_dim(3000, 500,dict_ranks, nb_max_iter=100000, batches_size=batches_size)
# batches_size = [20,50]
# plot_n_or_dim(210, 50,dict_ranks, nb_max_iter=1600, batches_size=batches_size)
