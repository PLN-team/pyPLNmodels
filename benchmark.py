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


def append_running_times_model(Y, model_str, n, time_limit, keep_going):
    if keep_going is True:
        if model_str == "Pln":
            model = Pln(Y)
            model.fit(nb_max_iteration=2000)
        else:
            model = PlnPCA(Y, rank=rank)
            model.fit(nb_max_iteration=2000)

        dict_rt[model_str][n].append(model._criterion_args.running_times[-1])
        if dict_rt[model_str][n][-1] > time_limit:
            keep_going = False
            print("Returning False")
            print("time", model._criterion_args.running_times[-1])
    else:
        dict_rt[model_str][n].append(None)
    return keep_going


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
            x=ps_last,
            y=res_model,
            color=color,
            label=f"{model_str}, n = {n}",
            linestyle=linestyles[i],
        )
    plt.legend()
    plt.xlabel(r"Number of variables $p$")
    plt.ylabel("Running time")


def dict_to_df(model_str):
    dict_dim = {"dim": ps_last}
    union_dict = {**dict_rt[model_str], **dict_dim}
    columns = ns
    columns.append("dim")
    return pd.DataFrame.from_dict(union_dict)


if __name__ == "__main__":
    n_data, p_data = 2000, 1500
    ns = [100, 1000, 1500]

    pmin = 5

    pn_first = 100
    ecart_first = 20

    pn_second = 300
    ecart_second = 100

    pn_third = 1000
    ecart_third = 200

    time_limit = 15

    ######

    # n_data, p_data = 20000, 15000
    # ns = [19000]

    # pmin = 5

    # pn_first = 200
    # ecart_first = 5

    # pn_second = 2000
    # ecart_second = 100

    # pn_third = 16000
    # ecart_third = 1000

    # time_limit = 10000

    if p_data == 15000:
        namefile = "full_scmark.csv"
    else:
        namefile = "full_scmark_little.csv"
    # full, _, _ = get_sc_mark_data(max_n=n_data, dim=p_data)
    # full = np.flip(full, axis=1)
    # df = pd.DataFrame(full)
    # df.to_csv(namefile, index=False)

    fig = plt.figure(figsize=(20, 10))

    ps_first = np.arange(pmin, pn_first, ecart_first).astype(int)
    ps_second = np.concatenate(
        (ps_first, np.arange(pn_first, pn_second, ecart_second))
    ).astype(int)
    ps_last = np.concatenate(
        (ps_second, np.arange(pn_second, pn_third, ecart_third))
    ).astype(int)
    print("ps last", ps_last)

    rank = 5
    df = pd.read_csv(namefile)

    dict_rt = {
        "Pln": {n: [] for n in ns},
        "PlnPCA": {n: [] for n in ns},
    }

    for n in ns:
        print("n:", n)
        keep_going_pln = True
        keep_going_plnpca = True
        name_file = f"dump/n_{n}_nbps_{len(ps_last)}_rank_{rank}"
        if True:
            # if exists(name_file) is False:
            for p in tqdm(ps_last):
                Y = df.values[:n, :p]
                # Y, _, labels = get_sc_mark_data(max_n=n, dim=p)
                keep_going_plnpca = append_running_times_model(
                    Y, "PlnPCA", n, time_limit, keep_going_plnpca
                )
                print("keep going pln", keep_going_pln)
                keep_going_pln = append_running_times_model(
                    Y, "Pln", n, time_limit, keep_going_pln
                )
                keep_going_plnpca = True
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
