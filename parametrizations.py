import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyPLNmodels import Pln, PlnPCA, Pln0
import seaborn as sns
from structpca import PlnPCAsampler, get_linear_params_and_additional_data
from structpca.pca_sampling_parameters import get_block_components
import jax.numpy as jnp
from jax import random
import jax.numpy.linalg as LA
from matplotlib.ticker import FormatStrFormatter

max_time = 60
nb_params = 5
n_samples = 1000
dim1 = 100
dim2 = 800
nb_cov = 2
parametrizations = ["Profiled", "uPLN-0", "uPLN-PF"]
colors = ["red", "blue", "green", "orange"]
nb_point = 15


def fit_models(seed_param, dim):
    mydict = get_linear_params_and_additional_data(
        seed=seed_param,
        n_samples=n_samples,
        nb_cov=nb_cov,
        dim=dim,
        latent_dimension=5,
    )
    mydict["exog"] = mydict["exog"].at[:, 0].set(1)
    C = mydict["components"]
    sigma = C @ (C.T)
    sigma = C @ (C.T) + jnp.eye(dim)
    mydict["components"] = LA.cholesky(sigma)
    simulator = PlnPCAsampler.from_dict(mydict)
    counts = simulator.sample(seed=0)
    dict_models = {}
    for parametrization in parametrizations:
        if parametrization == "0":
            pln = Pln0(
                np.asarray(counts),
                exog=np.asarray(simulator.exog),
                offsets=np.asarray(simulator.offsets),
                add_const=False,
            )
        elif parametrization == "PF":
            pln = PlnPCA(
                np.asarray(counts),
                exog=np.asarray(simulator.exog),
                offsets=np.asarray(simulator.offsets),
                rank=dim,
                add_const=False,
            )
        else:
            pln = Pln(
                np.asarray(counts),
                exog=np.asarray(simulator.exog),
                offsets=np.asarray(simulator.offsets),
                add_const=False,
            )
        pln.seed = seed_param
        pln.fit(max_time=max_time)
        dict_models[parametrization] = pln
    return dict_models


df = pd.DataFrame(
    columns=["model", "dim", "parametrization", "seed", "elbo", "time", "absc"]
)


def append_dict(dict_model):
    global df
    for parametrization in parametrizations:
        model = dict_model[parametrization]
        for time_number, i in enumerate(
            np.linspace(0, len(model.running_times) - 1, nb_point)
        ):
            i = int(i)
            df = df.append(
                {
                    "model": model.__class__.__name__,
                    "dim": model.dim,
                    "parametrization": parametrization,
                    "seed": model.seed,
                    "elbo": -model._elbos_list[i],
                    "time": model.running_times[i],
                    "absc": time_number,
                },
                ignore_index=True,
            )


def launch_dim(dim):
    for seed_param in range(nb_params):
        dict_plns = fit_models(seed_param, dim)
        dict_pcas = fit_models(seed_param, dim)
        append_dict(dict_plns)
        append_dict(dict_pcas)


fig, all_axes = plt.subplots(2, figsize=(20, 10), layout="constrained")

launch_dim(dim1)
launch_dim(dim2)
df.to_csv("df_parametrization.csv", index=False)

df = pd.read_csv("df_parametrization.csv").reset_index(drop=True)
nb_point = np.max(np.unique(df["absc"]))
nb_points_final = 5

parametrizations = ["Profiled", "uPLN-0", "uPLN-PF"]
colors = sns.color_palette("viridis")
my_palette = {"Profiled": colors[5], "uPLN-0": colors[0], "uPLN-PF": colors[1]}

df.loc[df["parametrization"] == "0", "parametrization"] = "uPLN-0"
df.loc[df["parametrization"] == "profiled", "parametrization"] = "Profiled"
df.loc[df["parametrization"] == "PF", "parametrization"] = "uPLN-PF"


dict_axes = {dim1: all_axes[0], dim2: all_axes[1]}

dict_bound = {dim1: (0.0, 100), dim2: (0, 60)}
for dim in [dim1, dim2]:
    df_ = df[df["dim"] == dim]
    for integer in np.unique(df_["absc"]):
        wanted = df_["absc"] == integer
        if dim == dim1:
            df_.loc[wanted, "time"] = np.round(np.mean(df_[wanted]["time"]), 3)
        else:
            df_.loc[wanted, "time"] = np.round(np.mean(df_[wanted]["time"]), 3)

    wanted_absc = (df_["time"] > dict_bound[dim][0]) & (
        df_["time"] < dict_bound[dim][1]
    )
    df_ = df_[wanted_absc]
    absc_unique = np.unique(df_["absc"])
    wanted_timepoint = (
        np.geomspace(max(np.min(absc_unique), 1), np.max(absc_unique), nb_points_final)
    ).astype(int)
    df_ = df_[df_["absc"].isin(wanted_timepoint)]
    sns.boxplot(
        x="time",
        y="elbo",
        data=df_,
        ax=dict_axes[dim],
        hue="parametrization",
        linewidth=0.5,
        showfliers=False,
        palette=my_palette,
    )


dict_axes[dim1].set_title(rf"Pln $p$={dim1}")
dict_axes[dim2].set_title(rf"Pln $p$={dim2}")


dict_axes[dim1].set_yscale("log")
dict_axes[dim2].set_yscale("log")

dict_axes[dim1].set_ylabel("Negative ELBO")
dict_axes[dim2].set_ylabel("Negative ELBO")
dict_axes[dim2].set_xlabel("Time (seconds)")

dict_axes[dim2].legend().remove()

handles, legend = all_axes[0].get_legend_handles_labels()
axbox = all_axes[1].get_position()

fig.legend(
    handles,
    legend,
    ncol=5,
    loc="lower center",
    fontsize=14,
    bbox_to_anchor=[0, axbox.y0 - 0.18, 1, 1],
    bbox_transform=fig.transFigure,
)

dict_axes[dim1].legend().remove()
# all_axes[0,0].legend()
plt.savefig(
    "/home/bastien/These/manuscript/tex/figures/intro/parametrizations.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
