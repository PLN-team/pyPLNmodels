from pyPLNmodels import Pln, PlnPCA
import scanpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


def get_sc_mark_data(max_class=28, max_n=200, dim=100):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:max_n]
    GT_name = data.obs["standard_true_celltype_v5"][:max_n]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    GT = GT[filter]
    GT_name = GT_name[filter]
    Y = Y[filter]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    var = np.var(Y, axis=0)
    most_variables = np.argsort(var)[-dim:]
    Y = Y[:, most_variables]
    return Y, GT, list(GT_name.values.__array__())


def append_running_times_pln_plnpca(Y):
    pln = Pln(Y)
    plnpca = PlnPCA(Y, rank=rank)
    pln.fit(tol=sharp_tol)
    plnpca.fit(tol=sharp_tol)

    rough_running_times_pln = pln._plotargs.running_times[
        next(i for i, v in enumerate(pln._plotargs.criterions) if v < rough_tol)
    ]
    rough_running_times_plnpca = plnpca._plotargs.running_times[
        next(i for i, v in enumerate(plnpca._plotargs.criterions) if v < rough_tol)
    ]

    dict_rt["pln_sharp"].append(pln._plotargs.running_times[-1])
    dict_rt["plnpca_sharp"].append(plnpca._plotargs.running_times[-1])
    dict_rt["pln_rough"].append(rough_running_times_pln)
    dict_rt["plnpca_rough"].append(rough_running_times_plnpca)


def plot_dict(dict_rt):
    plt.plot(ps, dict_rt["pln_sharp"], color="blue", label="Pln default tol")
    plt.plot(ps, dict_rt["plnpca_sharp"], color="orange", label="PlnPCA default tol")
    plt.plot(
        ps,
        dict_rt["pln_rough"],
        color="blue",
        label="Pln tol=0.01",
        linestyle="dotted",
    )
    plt.plot(
        ps,
        dict_rt["plnpca_rough"],
        color="orange",
        label="PlnPCA tol=0.01",
        linestyle="dotted",
    )
    plt.legend()
    plt.show()
    with open(name_file, "wb") as fp:
        pickle.dump(dict_rt, fp)


if __name__ == "__main__":
    n = 50
    p0 = 50
    pn = 100
    ecart = 10
    rank = 30
    ps = range(p0, pn, ecart)
    pln_running_times_sharp_conv = []
    plnpca_running_times_sharp_conv = []
    pln_running_times_rough_conv = []
    plnpca_running_times_rough_conv = []
    name_file = f"n_{n}_nbps_{len(ps)}_p0_{p0}_pn_{pn}_ecart_{ecart}_rank_{rank}"
    dict_rt = {
        "pln_sharp": pln_running_times_sharp_conv,
        "pln_rough": pln_running_times_rough_conv,
        "plnpca_sharp": plnpca_running_times_sharp_conv,
        "plnpca_rough": plnpca_running_times_rough_conv,
    }
    sharp_tol = 0.001
    rough_tol = 0.01
    if sharp_tol > rough_tol:
        raise ValueError("tols in the wrong order")

    if False:
        for p in ps:
            Y, _, _ = get_sc_mark_data(max_n=n, dim=p)
            append_running_times_pln_plnpca(Y)
    else:
        with open(name_file, "rb") as fp:
            dict_rt = pickle.load(fp)

    plot_dict(dict_rt)
