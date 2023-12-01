import scanpy
from pyPLNmodels.models import Pln, ZIPln
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


from sklearn.model_selection import cross_val_score


def get_sc_mark_data(max_class=28, max_n=200, dim=100, to_begin_with=None):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()
    GT_name = data.obs["standard_true_celltype_v5"]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    Y = Y[:, :dim]
    nb_non_zero = np.sum(Y > 0, axis=1)
    if to_begin_with is not None:
        # most_non_zeros = np.argsort(nb_non_zero)[11200:(11200+dim)]
        most_non_zeros = np.argsort(nb_non_zero)[
            to_begin_with : (to_begin_with + max_n)
        ]
    else:
        most_non_zeros = np.argsort(nb_non_zero)[-max_n:]
    Y = Y[most_non_zeros]
    GT = GT[most_non_zeros]
    GT_name = GT_name[most_non_zeros]
    le = LabelEncoder()
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
    return Y, GT, list(GT_name.values.__array__())


def launch_test():
    pass


cv = 4
nb_iter = 150
dims = [20, 50, 80]
# ps = [50, 75]#,100,125,150,200,300, 400, 500, 600, 700, 800, 900, 1000]
to_begins_with = [None, 19000, 18500, 18000, 17500]
n = 700
fig, axes = plt.subplots(len(dims), 1, figsize=(20, 20))
markers = {"green": "o", "orange": ">", "red": None, "black": "s"}


def launch_dim(dim, ax):
    zi_closed_multi_list = []
    zi_closed_single_list = []
    zi_free_multi_list = []
    zi_free_single_list = []
    zi_closed_n_list = []
    zi_free_n_list = []
    zi_closed_np_list = []
    zi_free_np_list = []
    pln_list = []
    absc = []

    def get_all_scores(n, p, to_begin_with):
        Y, GT_, _ = get_sc_mark_data(max_n=n, dim=p, to_begin_with=to_begin_with)
        percentage_zero = np.sum(Y == 0) / np.prod(Y.shape)
        print("percentage of zeros", percentage_zero)
        print(Y.shape)
        absc.append(percentage_zero)

        tol = 0.00

        def get_and_append_score(model, is_inflated, list_score, ground_truth):
            model.fit(tol=tol, nb_max_iteration=nb_iter)
            if is_inflated:
                Z, _ = model.transform()
            else:
                Z = model.transform()
            xgb = XGBClassifier()
            le = LabelEncoder()
            print("unique ground truth", np.unique(ground_truth))
            score = np.mean(
                cross_val_score(
                    xgb, Z, ground_truth, cv=cv, scoring="balanced_accuracy"
                )
            )
            list_score.append(score)
            print("model:", model._NAME, " score:", score)

        zi_closed_single = ZIPln(Y, use_closed_form_prob=True, do_single_inflation=True)
        zi_closed_multi = ZIPln(Y, use_closed_form_prob=True)
        zi_free_single = ZIPln(Y, use_closed_form_prob=False, do_single_inflation=True)
        zi_free_multi = ZIPln(Y, use_closed_form_prob=False)
        zi_closed_n = ZIPln(Y, use_closed_form_prob=True, do_n_inflation=True)
        zi_free_n = ZIPln(Y, use_closed_form_prob=False, do_n_inflation=True)
        zi_closed_np = ZIPln(Y, use_closed_form_prob=True, do_np_inflation=True)
        zi_free_np = ZIPln(Y, use_closed_form_prob=False, do_np_inflation=True)
        pln = Pln(Y)

        get_and_append_score(zi_closed_single, True, zi_closed_single_list, GT_)
        get_and_append_score(pln, False, pln_list, GT_)
        get_and_append_score(zi_closed_multi, True, zi_closed_multi_list, GT_)
        get_and_append_score(zi_free_multi, True, zi_free_multi_list, GT_)
        get_and_append_score(zi_free_single, True, zi_free_single_list, GT_)
        get_and_append_score(zi_closed_n, True, zi_closed_n_list, GT_)
        get_and_append_score(zi_free_n, True, zi_free_n_list, GT_)
        get_and_append_score(zi_closed_np, True, zi_closed_np_list, GT_)
        get_and_append_score(zi_free_np, True, zi_free_np_list, GT_)

    for to_begin_with in to_begins_with:
        get_all_scores(n, dim, to_begin_with)

    linewidth = 2
    ax.set_title(f"Comparison of scores on scMark dataset with n={n},p={dim}, d=1")

    ax.plot(
        absc,
        zi_closed_multi_list,
        label="ZI Closed multi",
        color="red",
        linewidth=linewidth,
        marker=markers["red"],
    )
    ax.plot(
        absc,
        zi_free_multi_list,
        label="ZI free multi",
        color="red",
        linestyle="--",
        linewidth=linewidth,
        marker=markers["red"],
    )

    ax.plot(
        absc,
        zi_closed_single_list,
        label="ZI Closed single",
        color="green",
        linestyle="solid",
        linewidth=linewidth,
        marker=markers["green"],
    )
    ax.plot(
        absc,
        zi_free_single_list,
        label="ZI free single",
        color="green",
        linestyle="--",
        linewidth=linewidth,
        marker=markers["green"],
    )

    ax.plot(
        absc,
        zi_closed_n_list,
        label="ZI Closed n",
        color="black",
        linestyle="solid",
        linewidth=linewidth,
        marker=markers["black"],
    )
    ax.plot(
        absc,
        zi_free_n_list,
        label="ZI free n",
        color="black",
        linestyle="--",
        linewidth=linewidth,
        marker=markers["black"],
    )

    ax.plot(
        absc,
        zi_closed_np_list,
        label="ZI Closed np",
        color="orange",
        linestyle="solid",
        linewidth=linewidth,
        marker=markers["orange"],
    )
    ax.plot(
        absc,
        zi_free_np_list,
        label="ZI free np",
        color="orange",
        linestyle="--",
        linewidth=linewidth,
        marker=markers["orange"],
    )

    ax.plot(
        absc,
        pln_list,
        label="pln",
        color="blue",
        linestyle="dotted",
        linewidth=linewidth,
    )

    ax.set_xlabel("percentage of zeros", fontsize=10)
    ax.set_ylabel(f"Xgb score with {cv} crossval", fontsize=20)


for i in range(len(dims)):
    dim_ = dims[i]
    ax_ = axes[i]
    launch_dim(dim_, ax_)


axes[0].legend()
plt.savefig(f"xgb_fit_pln_dims_{dims}.pdf", format="pdf")
plt.show()
