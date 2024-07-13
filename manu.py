import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyPLNmodels import Pln, PlnPCA
import seaborn as sns
from structpca import PlnPCAsampler, get_linear_params_and_additional_data


max_epoch = 400
max_time = 2
nb_params = 3
n_samples = 1000
dim1 = 20
dim2 = 30
nb_cov = 2
latent_dim = 5
nb_grad_steps = [0, 1, 10, 20]
colors = ["red", "blue", "green", "orange"]
nb_point = 4


def fit_models(model, seed_param, dim):
    mydict = get_linear_params_and_additional_data(
        seed=seed_param,
        n_samples=n_samples,
        nb_cov=nb_cov,
        dim=dim,
        latent_dimension=5,
    )
    if model == Pln:
        mydict["components"] *= 1

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
        if nb_grad_step == 0:
            pln.fit(max_time=max_time)
        else:
            pln.fit_vem(nb_gradient_steps=nb_grad_step, max_time=max_time)
        dict_models[nb_grad_step] = pln
    return dict_models


fig, all_axes = plt.subplots(2, 2, figsize=(20, 10), layout="constrained")


def plot_dict(dict_model, modeltype, ax, tolegend):
    for i in range(len(nb_grad_steps)):
        nb_grad_step = nb_grad_steps[i]
        model = dict_model[nb_grad_step]
        if tolegend is True:
            ax.plot(
                model.running_times,
                -np.array(model._elbos_list),
                label=nb_grad_step,
                color=colors[i],
            )
        else:
            ax.plot(model.running_times, -np.array(model._elbos_list), color=colors[i])


df = pd.DataFrame(columns=["model", "dim", "nb_grad_steps", "seed", "elbo", "time"])


def append_dict(dict_model):
    global df
    for nb_grad_step in nb_grad_steps:
        model = dict_model[nb_grad_step]
        for i in np.linspace(0, len(model.running_times), nb_point):
            i = int(i)
            print("i:", i)
            print("len", len(model.running_times))
            df = df.append(
                {
                    "model": model.__class__.__name__,
                    "dim": model.dim,
                    "nb_grad_steps": nb_grad_step,
                    "seed": model.seed,
                    "elbo": -model._elbos_list[i],
                    "time": model.running_times[i],
                },
                ignore_index=True,
            )


def launch_dim(dim, axes):
    for seed_param in range(nb_params):
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


launch_dim(dim1, [all_axes[0, 0], all_axes[0, 1]])
# launch_dim(dim2, [all_axes[1, 0], all_axes[1, 1]])
# all_axes[0, 0].legend()
print("df :", df)
plt.show()
