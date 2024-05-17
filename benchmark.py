from os.path import exists
import os
from pyPLNmodels import Pln, PlnPCA
import numpy as np
import pandas as pd
import scanpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import torch


if torch.cuda.is_available():
    DEVICE = "GPU"
else:
    DEVICE = "cpu"


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


def append_running_times_model(Y, model_str, n):
    if model_str == "Pln":
        model = Pln(Y)
        model.fit(nb_max_iteration=2000)
    else:
        model = PlnPCA(Y, rank=rank)
        model.fit(nb_max_iteration=2000)
    dict_rt[model_str][n].append(model._criterion_args.running_times[-1])


def plot_dict(dict_rt, model_str, ns):
    dict_rt_model = dict_rt[str(model_str)]
    linestyles = ["--", "solid", "dashed", "dashdot"]
    for i, n in enumerate(ns):
        res_model = dict_rt_model[n]
        if model_str == "Pln":
            color = "blue"
        else:
            color = "orange"
        sns.lineplot(
            x=ps_plnpca,
            y=res_model,
            color=color,
            label=f"{model_str}, n = {n}",
            linestyle=linestyles[i],
        )
    plt.legend()
    plt.xlabel(r"Number of variables $p$")
    plt.ylabel("Running time")


def dict_to_df(model_str):
    dict_dim = {"dim": ps_plnpca}
    union_dict = {**dict_rt[model_str], **dict_dim}
    columns = ns
    columns.append("dim")
    return pd.DataFrame.from_dict(union_dict)


if __name__ == "__main__":
    # n_data, p_data = 2000,1500
    # ns = [100, 1000,1500]
    # pmin = 5
    # pn_first = 100
    # ecart_first = 50
    # rank = 5
    # ecart_second = 100
    # pmax_pln = 300
    # pn_second = pn_first + 300

    n_data, p_data = 20000, 15000
    ns = [19000]
    pmin = 5
    pn_first = 100
    ecart_first = 50
    rank = 5
    ecart_second = 2000
    pmax_pln = 300
    pn_second = 15000

    if p_data == 15000:
        namefile = "full_scmark.csv"
    else:
        namefile = "full_scmark_little.csv"
    full, _, _ = get_sc_mark_data(max_n=n_data, dim=p_data)
    full = np.flip(full, axis=1)
    df = pd.DataFrame(full)
    df.to_csv(namefile, index=False)
    df = pd.read_csv(namefile)

    fig = plt.figure(figsize=(20, 10))

    ps_pln_first = np.arange(pmin, pn_first, ecart_first).astype(int)
    ps_pln = np.concatenate(
        (ps_pln_first, np.arange(pn_first, pmax_pln, ecart_second))
    ).astype(int)
    ps_plnpca = np.concatenate(
        (ps_pln, np.arange(pn_first, pn_second, ecart_second))
    ).astype(int)
    print("ps_plnpca", ps_plnpca)

    dict_rt = {
        "Pln": {n: [] for n in ns},
        "PlnPCA": {n: [] for n in ns},
    }

    for n in ns:
        print("n:", n)
        name_file = f"dump/n_{n}_nbps_{len(ps_plnpca)}_p0_{pmax_pln}_rank_{rank}"
        if True:
            # if exists(name_file) is False:
            for p in tqdm(ps_plnpca):
                Y = df.values[:n, :p]
                # Y, _, labels = get_sc_mark_data(max_n=n, dim=p)
                append_running_times_model(Y, "PlnPCA", n)
                if p < pmax_pln:
                    append_running_times_model(Y, "Pln", n)
                else:
                    dict_rt["Pln"][n].append(None)

        else:
            with open(name_file, "rb") as fp:
                dict_rt = pickle.load(fp)

        with open(name_file, "wb") as fp:
            pickle.dump(dict_rt, fp)

    plot_dict(dict_rt, "Pln", ns)
    plot_dict(dict_rt, "PlnPCA", ns)
    df_results_pln = dict_to_df("Pln")
    print("res pln", df_results_pln)
    df_results_plnpca = dict_to_df("PlnPCA")
    df_results_pln.to_csv(f"csv_res_benchmark/python_pln_{DEVICE}.csv")
    df_results_plnpca.to_csv(f"csv_res_benchmark/python_plnpca_{DEVICE}.csv")
    plt.savefig(f"paper/illustration.png", format="png")
    plt.show()
