import time
import math
import scanpy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from pyPLNmodels._utils import sample_PLN
from pyPLNmodels import PLNPCA, PLN
import pandas as pd
from tests.utils import get_real_data
from sklearn import svm
from umap import UMAP
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# true_beta = torch.randn(d, p, device=DEVICE)
# C = torch.randn(p, q, device=DEVICE)/5
# O = torch.ones((n, p), device=DEVICE)/2
# covariates = torch.ones((n, d), device=DEVICE)*0 + 1
# true_Sigma = torch.matmul(C,C.T)
# Y, _, _ = sample_PLN(C, true_beta, covariates, O)
# Y = pd.read_csv("./example_data/test_data/Y_test.csv")
# covariates = pd.read_csv("./example_data/test_data/cov_test.csv")
# O = pd.read_csv("./example_data/test_data/O_test.csv")
# true_Sigma = torch.from_numpy(
#     pd.read_csv("./example_data/test_data/true_parameters/true_Sigma_test.csv").values
# )
# true_beta = torch.from_numpy(
#     pd.read_csv("./example_data/test_data/true_parameters/true_beta_test.csv").values
# )
# Y = pd.read_csv("./example_data/real_data/oaks_counts.csv")
# covariates = None
# O = np.log(pd.read_csv("./example_data/real_data/oaks_offsets.csv"))


def get_test_accuracy(X, y):
    xgb = XGBClassifier()
    svmclf = svm.SVC()
    if isinstance(X, torch.Tensor):
        X = X.cpu()
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    score_xgb = np.mean(cross_val_score(xgb, X, y, cv=cv, scoring="balanced_accuracy"))
    score_svm = np.mean(
        cross_val_score(svmclf, X, y, cv=cv, scoring="balanced_accuracy")
    )
    return {"xgb": score_xgb, "svm": score_svm}


def log_normalization(Y):
    return np.log(Y + (Y == 0) * math.exp(-2))


def test_dimension(max_dim, plot=False):
    Y, GT = get_real_data(take_oaks=False, max_n=n, max_class=8, max_dim=max_dim)

    ## log normalization
    lognorm_score = get_test_accuracy(log_normalization(Y), GT)

    #### pca
    pca = PLNPCA(ranks=RANKS)
    pca.fit(Y, O_formula="sum")
    latent_pca_first = pca[RANKS[0]].latent_variables
    pca_score_first = get_test_accuracy(latent_pca_first, GT)

    latent_pca_second = pca[RANKS[1]].latent_variables
    pca_score_second = get_test_accuracy(latent_pca_second, GT)

    latent_pca_proj = pca[RANKS[1]].projected_latent_variables
    pca_score_proj = get_test_accuracy(latent_pca_proj, GT)

    ### pln with sum formula for O
    pln = PLN()
    pln.fit(Y, O_formula="sum")
    latent = pln.latent_variables
    pln_score = get_test_accuracy(latent, GT)

    ### pln without sum formula
    plnzero = PLN()
    plnzero.fit(Y, O_formula=None)
    latent_zero = plnzero.latent_variables
    plnzero_score = get_test_accuracy(latent_zero, GT)

    if plot is True:
        dr = UMAP()

        drlogY = dr.fit_transform(Y)
        drlatent_pca_first = dr.fit_transform(latent_pca_first)
        drlatent_pca_second = dr.fit_transform(latent_pca_second)
        drlatent = dr.fit_transform(latent)
        drlatent_zero = dr.fit_transform(latent_zero)

        fig, axes = plt.subplots(5)
        axes[0].scatter(drlogY[:, 0], drlogY[:, 1], c=GT)
        axes[0].legend()
        axes[0].set_title("UMAP after log normalization")

        axes[1].scatter(drlatent[:, 0], drlatent[:, 1], c=GT)
        axes[1].legend()
        axes[1].set_title("UMAP after normalization with pln")

        axes[2].scatter(drlatent_pca_first[:, 0], drlatent_pca_first[:, 1], c=GT)
        axes[2].legend()
        axes[2].set_title(f"UMAP after normalization with plnpca with rank{RANKS[0]}")

        axes[3].scatter(drlatent_pca_second[:, 0], drlatent_pca_second[:, 1], c=GT)
        axes[3].legend()
        axes[3].set_title(f"UMAP after normalization with plnpca with rank{RANKS[1]}")

        axes[4].scatter(drlatent_zero[:, 0], drlatent_zero[:, 1], c=GT)
        axes[4].legend()
        axes[4].set_title("UMAP after normalization with pln zero")
        plt.show()
    return (
        lognorm_score,
        pca_score_first,
        pca_score_second,
        pca_score_proj,
        pln_score,
        plnzero_score,
    )


def append_scores(method, new_score):
    method["xgb"].append(new_score["xgb"])
    method["svm"].append(new_score["svm"])


def test_dimensions(max_dims, plot=False):

    scores_lognorm = {"xgb": [], "svm": [], "name": "lognorm", "linestyle": "-"}
    scores_pca_first = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca{RANKS[0]}",
        "linestyle": "--",
    }
    scores_pca_second = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca{RANKS[1]}",
        "linestyle": "dotted",
    }
    scores_pca_proj = {
        "xgb": [],
        "svm": [],
        "name": f"plnpca_projected_dim{RANKS[1]}",
        "linestyle": (5, (10, 3)),
    }
    scores_pln = {"xgb": [], "svm": [], "name": "pln", "linestyle": "dashdot"}
    scores_plnzero = {
        "xgb": [],
        "svm": [],
        "name": "plnzero",
        "linestyle": (0, (1, 10)),
    }

    for max_dim in max_dims:
        print("Dimension :", max_dim)
        (
            new_lognorm_score,
            new_pca_score_first,
            new_pca_score_second,
            new_pca_score_proj,
            new_pln_score,
            new_plnzero_score,
        ) = test_dimension(max_dim, plot)
        append_scores(scores_lognorm, new_lognorm_score)
        append_scores(scores_pca_first, new_pca_score_first)
        append_scores(scores_pca_second, new_pca_score_second)
        append_scores(scores_pca_proj, new_pca_score_proj)
        append_scores(scores_pln, new_pln_score)
        append_scores(scores_plnzero, new_plnzero_score)

    return [
        scores_lognorm,
        scores_pca_first,
        scores_pca_second,
        scores_pca_proj,
        scores_pln,
        scores_plnzero,
    ]


def plot_res(res, dims):
    fig, ax = plt.subplots()
    for score in res:
        label = score["name"]
        to_plot_xgb = list(score["xgb"])
        to_plot_svm = list(score["svm"])
        ax.plot(
            dims,
            to_plot_xgb,
            label=label + "xgb",
            color="blue",
            linestyle=score["linestyle"],
        )
        ax.plot(
            dims,
            to_plot_svm,
            label=label + "svm",
            color="red",
            linestyle=score["linestyle"],
        )
        ax.legend()
    plt.show()


RANKS = [10, 80]
cv = 10
n = 5000
max_dims = [
    80,
    100,
    150,
    250,
    400,
    600,
    800,
    1000,
    1300,
    1500,
    1800,
    2000,
    2500,
    3000,
    4000,
]

need_to_compute = False
file_name = f"n={n}cv={cv}"
if need_to_compute is True:
    res = test_dimensions(max_dims=max_dims, plot=False)
    with open(file_name, "wb") as fp:
        pickle.dump(res, fp)
else:
    with open(file_name, "rb") as fp:
        res = pickle.load(fp)

plot_res(res, max_dims)
