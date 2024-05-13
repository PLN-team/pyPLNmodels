from os.path import exists
from pyPLNmodels import Pln, PlnPCA
import numpy as np
import scanpy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os
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
    if model_str == "Pln":
        model = Pln(Y)
        model.fit(nb_max_iteration=2000)
        model.show()
        model_batch = Pln(Y, batch_size=batch_size)
        model_batch.fit(nb_max_iteration=1500, verbose=True)
        model_batch.show()
        x
    else:
        model = PlnPCA(Y, rank=rank)
        model.fit(nb_max_iteration=2000, tol=0)
        model.show()
        model_batch = PlnPCA(Y, batch_size=batch_size, rank=rank)
        model_batch.fit(nb_max_iteration=1500, verbose=True, tol=0)
        model_batch.show()

    dict_rt[model_str]["batch"].append(model._criterion_args.running_times[-1])
    dict_rt[model_str]["no batch"].append(model_batch._criterion_args.running_times[-1])


def plot_dict(dict_rt, model_str):
    print("dict_rt", dict_rt)
    print("model str", str(model_str))
    dict_rt_model = dict_rt[str(model_str)]
    if model_str == "Pln":
        ps = ps_pln
        color = "blue"
    else:
        ps = ps_plnpca
        color = "orange"
    print("ps_pln:", ps_pln)
    sns.lineplot(x=ps, y=dict_rt_model["batch"], color=color, label=f"{model_str}")
    sns.lineplot(
        x=ps,
        y=dict_rt_model["no batch"],
        color=color,
        label=f"{model_str} batch",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel(r"Number of variables $p$")
    plt.ylabel("Running time")


if __name__ == "__main__":
    n = 2000
    batch_size = 100
    # p0 = 2500
    # pn = 14059
    # ecart = 300
    fig = plt.figure(figsize=(20, 10))
    pmax_pln = 300
    pn = 500
    ecart = 100
    rank = 40
    ps_pln = np.arange(100, pmax_pln, ecart)
    ps_plnpca = np.concatenate((ps_pln, np.arange(pmax_pln, pn, ecart)))
    name_file = (
        f"n_{n}_nbps_{len(ps_plnpca)}_p0_{pmax_pln}_pn_{pn}_ecart_{ecart}_rank_{rank}"
    )
    dict_rt = {
        "Pln": {"batch": [], "no batch": []},
        "PlnPCA": {"batch": [], "no batch": []},
    }

    if True:
        # if exists(name_file) is False:
        for p in tqdm(ps_plnpca):
            print("dim:", p)
            Y, _, labels = get_sc_mark_data(max_n=n, dim=p)
            append_running_times_model(Y, "PlnPCA")
            if p < pmax_pln:
                append_running_times_model(Y, "Pln")
    else:
        with open(name_file, "rb") as fp:
            dict_rt = pickle.load(fp)

    with open(name_file, "wb") as fp:
        pickle.dump(dict_rt, fp)
    plot_dict(dict_rt, "Pln")
    plot_dict(dict_rt, "PlnPCA")
    plt.savefig(f"paper/illustration.png", format="png")
    plt.show()
