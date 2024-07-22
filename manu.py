import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyPLNmodels import PlnPCA
import seaborn as sns
from structpca import PlnPCAsampler, get_linear_params_and_additional_data
from structpca.pca_sampling_parameters import get_block_components
import jax.numpy as jnp
from jax import random
import jax.numpy.linalg as LA
from matplotlib.ticker import FormatStrFormatter

max_time = 120
nb_params = 15
n_samples = 1000
dim1 = 100
dim2 = 800
nb_cov = 2
latent_dim = 5
nb_grad_steps = ["Singular", 10, 30]
colors = ["red", "blue", "green", "orange"]
nb_point = 1000


sns.color_palette("viridis")


def fit_models(model, seed_param, dim):
    mydict = get_linear_params_and_additional_data(
        seed=seed_param,
        n_samples=n_samples,
        nb_cov=nb_cov,
        dim=dim,
        latent_dimension=latent_dim,
    )
    mydict["exog"] = mydict["exog"].at[:, 0].set(1)
    simulator = PlnPCAsampler.from_dict(mydict)
    counts = simulator.sample(seed=0)
    dict_models = {}
    for nb_grad_step in nb_grad_steps:
        pln = model(
            np.asarray(counts),
            exog=np.asarray(simulator.exog),
            offsets=np.asarray(simulator.offsets),
            rank=latent_dim,
            add_const=False,
        )
        pln.seed = seed_param
        if nb_grad_step == "Singular":
            pln.fit(max_time=max_time)
        else:
            pln.fit_vem(nb_gradient_steps=nb_grad_step, max_time=max_time)
        dict_models[nb_grad_step] = pln
    return dict_models


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
                dict_new = (
                    {
                        "model": model.__class__.__name__,
                        "dim": model.dim,
                        "nb_grad_steps": nb_grad_step,
                        "seed": model.seed,
                        "elbo": -model._elbos_list[i],
                        "time": model.running_times[i],
                        "absc": time_number,
                    },
                )
                new_df = pd.DataFrame.from_dict(dict_new)
                df = pd.concat([df, new_df], ignore_index=True)


def launch_dim(dim):
    for seed_param in range(nb_params):
        print("seed", seed_param)
        dict_pcas = fit_models(PlnPCA, seed_param, dim)
        append_dict(dict_pcas)


fig, all_axes = plt.subplots(2, figsize=(20, 10), layout="constrained")

launch_dim(dim1)
launch_dim(dim2)
df.to_csv("df.csv", index=False)

df = pd.read_csv("df.csv").reset_index(drop=True)
# df.loc[df["nb_grad_steps"] == '1',"nb_grad_steps"] = "Singular"
df.loc[df["nb_grad_steps"] == "10", "nb_grad_steps"] = r"$10$ gradient steps"
df.loc[df["nb_grad_steps"] == "30", "nb_grad_steps"] = r"$30$ gradient steps"
df["model"] = "PLN-PCA"
nb_grad_steps = ["Singular", r"$10$ gradient steps", r"$30$ gradient steps"]
colors = sns.color_palette("viridis")

my_palette = {
    r"$10$ gradient steps": colors[1],
    r"$30$ gradient steps": colors[0],
    "Singular": colors[2],
}
# my_palette = {nb_grad_steps[i]: colors[i] for i in range(4)}

nb_point = np.max(np.unique(df["absc"]))
nb_points_final = 12


dict_axes = {
    dim1: {"PLN-PCA": all_axes[0]},
    dim2: {"PLN-PCA": all_axes[1]},
}
dict_bound = {
    dim1: {"PLN-PCA": (0.1, 50)},
    dim2: {"PLN-PCA": (0.1, 50)},
}
for dim in [dim1, dim2]:
    model = "PLN-PCA"
    df_ = df[(df["dim"] == dim) & (df["model"] == model)]
    print("df before:", df_)
    for integer in np.unique(df_["absc"]):
        wanted = df_["absc"] == integer
        df_.loc[wanted, "time"] = np.round(np.mean(df_[wanted]["time"]), 1)
    wanted_absc = (df_["time"] > dict_bound[dim][model][0]) & (
        df_["time"] < dict_bound[dim][model][1]
    )
    df_ = df_[wanted_absc]
    absc_unique = np.unique(df_["absc"])

    wanted_timepoint = (
        np.geomspace(max(np.min(absc_unique), 1), np.max(absc_unique), nb_points_final)
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


dict_axes[dim1]["PLN-PCA"].set_title(rf"PLN-PCA $p$={dim1}", fontsize=18)
dict_axes[dim2]["PLN-PCA"].set_title(rf"PLN-PCA $p$={dim2}", fontsize=18)


dict_axes[dim1]["PLN-PCA"].set_yscale("log")
dict_axes[dim2]["PLN-PCA"].set_yscale("log")

dict_axes[dim1]["PLN-PCA"].set_ylabel("Negative ELBO", fontsize=18)
dict_axes[dim2]["PLN-PCA"].set_ylabel("Negative ELBO", fontsize=18)
dict_axes[dim2]["PLN-PCA"].set_xlabel("Time (seconds)", fontsize=18)
dict_axes[dim1]["PLN-PCA"].set_xlabel("")


dict_axes[dim1]["PLN-PCA"].tick_params(axis="both", which="major", labelsize=20)
dict_axes[dim1]["PLN-PCA"].tick_params(axis="both", which="minor", labelsize=12)
dict_axes[dim2]["PLN-PCA"].tick_params(axis="both", which="major", labelsize=20)
dict_axes[dim2]["PLN-PCA"].tick_params(axis="both", which="minor", labelsize=12)


# dict_axes[dim2]["Pln"].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# dict_axes[dim2]["PLN-PCA"].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# dict_axes[dim1]["PLN-PCA"].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# dict_axes[dim2]["Pln"].legend().remove()
dict_axes[dim1]["PLN-PCA"].legend().remove()
dict_axes[dim2]["PLN-PCA"].legend().remove()

handles, legend = all_axes[0].get_legend_handles_labels()
axbox = all_axes[1].get_position()

fig.legend(
    handles,
    legend,
    ncol=5,
    loc="lower center",
    fontsize=18,
    bbox_to_anchor=[0, axbox.y0 - 0.18, 1, 1],
    bbox_transform=fig.transFigure,
)


# dict_axes[dim1]["PLN-PCA"].legend().remove()
plt.savefig(
    "/home/bastien/These/manuscript/tex/figures/intro/elbos.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

# if len(df_) > 0:
# plt.plot(df_["time"], df_["elbo"], label=f"{model} {nb_grad_step}")
