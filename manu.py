import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyPLNmodels import Pln, PlnPCA
import seaborn as sns
from structpca import PlnPCAsampler, get_linear_params_and_additional_data
from structpca.pca_sampling_parameters import get_block_components
import jax.numpy as jnp
from jax import random
import jax.numpy.linalg as LA
from matplotlib.ticker import FormatStrFormatter

max_time = 60
nb_params = 15
n_samples = 1000
dim1 = 100
dim2 = 800
nb_cov = 2
latent_dim = 5
nb_grad_steps = ["profiled", 1, 10, 30]
colors = ["red", "blue", "green", "orange"]
nb_point = 2000


colors = sns.color_palette("viridis")
my_palette = {"profiled": colors[0], "1": colors[1], "10": colors[2], "30": colors[3]}
sns.color_palette("viridis")


def fit_models(model, seed_param, dim):
    mydict = get_linear_params_and_additional_data(
        seed=seed_param,
        n_samples=n_samples,
        nb_cov=nb_cov,
        dim=dim,
        latent_dimension=5,
    )
    if model == Pln:
        C = mydict["components"]
        sigma = C @ (C.T)
        sigma = C @ (C.T) + jnp.eye(dim)
        mydict["components"] = LA.cholesky(sigma)
    simulator = PlnPCAsampler.from_dict(mydict)
    counts = simulator.sample(seed=0)
    dict_models = {}
    for nb_grad_step in nb_grad_steps:
        if model == PlnPCA:
            pln = model(
                np.asarray(counts),
                exog=np.asarray(simulator.exog),
                offsets=np.asarray(simulator.offsets),
                rank=latent_dim,
                add_const=False,
            )
        else:
            pln = model(
                np.asarray(counts),
                exog=np.asarray(simulator.exog),
                offsets=np.asarray(simulator.offsets),
                add_const=False,
            )
        pln.seed = seed_param
        if nb_grad_step == "profiled":
            if model == Pln:
                pln.fit(max_time=max_time)
            else:
                pln = None
        else:
            if model == Pln:
                pln.fit_vem(nb_gradient_steps=nb_grad_step, max_time=max_time)
            else:
                pln.fit_vem(nb_gradient_steps=nb_grad_step, max_time=2 * max_time)
        dict_models[nb_grad_step] = pln
    return dict_models


# def plot_dict(dict_model, modeltype, ax, tolegend):
#     for i in range(len(nb_grad_steps)):
#         nb_grad_step = nb_grad_steps[i]
#         model = dict_model[nb_grad_step]
#         if tolegend is True:
#             ax.plot(
#                 model.running_times,
#                 -np.array(model._elbos_list),
#                 label=nb_grad_step,
#                 color=colors[i],
#             )
#         else:
#             ax.plot(model.running_times, -np.array(model._elbos_list), color=colors[i])


df = pd.DataFrame(
    columns=["model", "dim", "nb_grad_steps", "seed", "elbo", "time", "absc"]
)


def append_dict(dict_model):
    global df
    for nb_grad_step in nb_grad_steps:
        model = dict_model[nb_grad_step]
        if model is not None:
            for time_number, i in enumerate(
                np.linspace(0, len(model.running_times) - 1, nb_point)
            ):
                i = int(i)
                df = df.append(
                    {
                        "model": model.__class__.__name__,
                        "dim": model.dim,
                        "nb_grad_steps": nb_grad_step,
                        "seed": model.seed,
                        "elbo": -model._elbos_list[i],
                        "time": model.running_times[i],
                        "absc": time_number,
                    },
                    ignore_index=True,
                )


def launch_dim(dim, axes):
    for seed_param in range(nb_params):
        print("seed", seed_param)
        dict_plns = fit_models(Pln, seed_param, dim)
        dict_pcas = fit_models(PlnPCA, seed_param, dim)
        append_dict(dict_plns)
        append_dict(dict_pcas)
        if seed_param == 0 and dim == dim1:
            tolegend = True
        else:
            tolegend = False
        # plot_dict(dict_plns, "Pln", axes[0], tolegend=tolegend)
        # plot_dict(dict_pcas, "PlnPCA", axes[1], tolegend=False)
        # axes[0].set_yscale("log")


fig, all_axes = plt.subplots(2, 2, figsize=(20, 10), layout="constrained")

# launch_dim(dim1, [all_axes[0, 0], all_axes[0, 1]])
# launch_dim(dim2, [all_axes[1, 0], all_axes[1, 1]])
# df.to_csv("df.csv", index = False)

df = pd.read_csv("df.csv").reset_index(drop=True)
df = df.iloc[::-1]
nb_point = np.max(np.unique(df["absc"]))
nb_points_final = 8


dict_axes = {
    dim1: {"Pln": all_axes[0, 0], "PlnPCA": all_axes[0, 1]},
    dim2: {"Pln": all_axes[1, 0], "PlnPCA": all_axes[1, 1]},
}
dict_bound = {
    dim1: {"Pln": (0.01, 0.76), "PlnPCA": (1, 120)},
    dim2: {"Pln": (3, 60), "PlnPCA": (5, 120)},
}
# print('df', df[(df["dim"] == dim2) & (df["model"] == "PlnPCA")]["time"][0:50] )
for dim in [dim1, dim2]:
    for model in ["Pln", "PlnPCA"]:
        df_ = df[(df["dim"] == dim) & (df["model"] == model)]
        for integer in np.unique(df_["absc"]):
            wanted = df_["absc"] == integer
            if model == "Pln" and dim == dim1:
                df_.loc[wanted, "time"] = np.round(np.mean(df_[wanted]["time"]), 2)
            else:
                df_.loc[wanted, "time"] = np.round(
                    np.mean(df_[wanted]["time"]), 1
                )  # .astype(int)
                # print('time', np.unique(df_[wanted]["time"]))
                # n_round = 0

        wanted_absc = (df_["time"] > dict_bound[dim][model][0]) & (
            df_["time"] < dict_bound[dim][model][1]
        )
        df_ = df_[wanted_absc]
        absc_unique = np.unique(df_["absc"])
        wanted_timepoint = (
            np.geomspace(
                max(np.min(absc_unique), 1), np.max(absc_unique), nb_points_final
            )
        ).astype(int)
        df_ = df_[df_["absc"].isin(wanted_timepoint)]
        print("unique", np.unique(df_["nb_grad_steps"]))
        sns.boxplot(
            x="time",
            y="elbo",
            data=df_,
            ax=dict_axes[dim][model],
            hue="nb_grad_steps",
            linewidth=0.5,
            showfliers=False,
            palette=my_palette,
        )


dict_axes[dim1]["Pln"].set_title(rf"Pln $p$={dim1}")
dict_axes[dim1]["PlnPCA"].set_title(rf"PlnPCA $p$={dim1}")
dict_axes[dim2]["Pln"].set_title(rf"Pln $p$={dim2}")
dict_axes[dim2]["PlnPCA"].set_title(rf"PlnPCA $p$={dim2}")


dict_axes[dim1]["Pln"].set_yscale("log")
dict_axes[dim2]["Pln"].set_yscale("log")
dict_axes[dim1]["PlnPCA"].set_yscale("log")
dict_axes[dim2]["PlnPCA"].set_yscale("log")

dict_axes[dim1]["Pln"].set_ylabel("Negative ELBO")
dict_axes[dim2]["Pln"].set_ylabel("Negative ELBO")
dict_axes[dim2]["Pln"].set_xlabel("Time (seconds)")
dict_axes[dim2]["PlnPCA"].set_xlabel("Time (seconds)")

# dict_axes[dim2]["Pln"].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# dict_axes[dim2]["PlnPCA"].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# dict_axes[dim1]["PlnPCA"].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

dict_axes[dim2]["Pln"].legend().remove()
dict_axes[dim1]["PlnPCA"].legend().remove()
dict_axes[dim2]["PlnPCA"].legend().remove()

handles, legend = all_axes[0, 0].get_legend_handles_labels()
axbox = all_axes[1, 0].get_position()

fig.legend(
    handles,
    legend,
    ncol=5,
    loc="lower center",
    fontsize=14,
    bbox_to_anchor=[0, axbox.y0 - 0.18, 1, 1],
    bbox_transform=fig.transFigure,
)

# fig.legend(handles, legend, ncol = 4, loc = 'lower center', fontsize = 14)#,bbox_transform=fig.transFigure)

dict_axes[dim1]["Pln"].legend().remove()
# all_axes[0,0].legend()
plt.savefig(
    "/home/bastien/These/manuscript/tex/figures/intro/elbos.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

# if len(df_) > 0:
# plt.plot(df_["time"], df_["elbo"], label=f"{model} {nb_grad_step}")
