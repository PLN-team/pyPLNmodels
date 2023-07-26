from pyPLNmodels import Pln, PlnPCA
import scanpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm


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


def append_running_times_model(Y, model_str):
    if model_str == "pln":
        model = Pln(Y)
        model.fit(tol=sharp_tol)
    else:
        model = PlnPCA(Y, rank=rank)
        model.fit(tol=sharp_tol)

    rough_running_times = model._plotargs.running_times[
        next(i for i, v in enumerate(model._plotargs.criterions) if v < rough_tol)
    ]

    dict_rt[model_str]["rough"].append(rough_running_times)
    dict_rt[model_str]["sharp"].append(model._plotargs.running_times[-1])


def plot_dict(dict_rt, model_str):
    dict_rt_model = dict_rt[model_str]
    if model_str == "pln":
        ps = ps_pln
        color = "blue"
    else:
        ps = ps_plnpca
        color = "orange"
    print("ps_pln:", ps_pln)
    print("dict model ", dict_rt_model["sharp"])
    plt.plot(ps, dict_rt_model["sharp"], color=color, label=f"{model_str} default tol")
    plt.plot(
        ps,
        dict_rt_model["rough"],
        color=color,
        label=f"{model_str} tol=0.01",
        linestyle="dotted",
    )
    plt.legend()
    plt.xlabel(r"Number of variables $p$")
    plt.ylabel("Running times")


if __name__ == "__main__":
    n = 200
    p0 = 2500
    pn = 14059
    ecart = 300
    fig = plt.figure(figsize=(20, 10))
    rank = 80
    ps_pln = np.arange(100, p0, 100)
    ps_plnpca = np.concatenate((ps_pln, np.arange(p0, pn, ecart)))
    pln_running_times_sharp_conv = []
    plnpca_running_times_sharp_conv = []
    pln_running_times_rough_conv = []
    plnpca_running_times_rough_conv = []
    name_file = f"n_{n}_nbps_{len(ps_plnpca)}_p0_{p0}_pn_{pn}_ecart_{ecart}_rank_{rank}"
    dict_rt = {
        "pln": {
            "sharp": pln_running_times_sharp_conv,
            "rough": pln_running_times_rough_conv,
        },
        "plnpca": {
            "sharp": plnpca_running_times_sharp_conv,
            "rough": plnpca_running_times_rough_conv,
        },
    }
    sharp_tol = 0.001
    rough_tol = 0.01
    if sharp_tol > rough_tol:
        raise ValueError("tols in the wrong order")

    if True:
        for p in tqdm(ps_plnpca):
            print("dim:", p)
            Y, _, _ = get_sc_mark_data(max_n=n, dim=p)
            append_running_times_model(Y, "plnpca")
            if p < p0:
                append_running_times_model(Y, "pln")
    else:
        with open(name_file, "rb") as fp:
            dict_rt = pickle.load(fp)

    plot_dict(dict_rt, "pln")
    plot_dict(dict_rt, "plnpca")
    plt.savefig(f"paper/illustration.png", format="png")
    plt.show()
    with open(name_file, "wb") as fp:
        pickle.dump(dict_rt, fp)
