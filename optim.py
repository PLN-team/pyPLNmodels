import torch
import numpy as np
from pyPLNmodels import load_scrna, Pln, PlnPCA
import matplotlib.pyplot as plt
import seaborn as sns

max_iter = 200
n_samples = 1000
dim = 1000
tol = 0.1

data = load_scrna(n_samples, dim)

colors = sns.color_palette("viridis")


fig, axes = plt.subplots(2, layout="constrained", figsize=(20, 10))

list_optim = [
    torch.optim.Adam,
    torch.optim.SGD,
    torch.optim.Rprop,
    torch.optim.RMSprop,
    torch.optim.Adagrad,
    torch.optim.Adamax,
]
name_optim = {
    torch.optim.Adam: "Adam",
    torch.optim.SGD: "SGD",
    torch.optim.Rprop: "Rprop",
    torch.optim.RMSprop: "RMSprop",
    torch.optim.Adagrad: "Adagrad",
    torch.optim.Adamax: "Adamax",
}

list_colors = [colors[i] for i in range(len(list_optim))]
my_palette = dict(zip(list_optim, list_colors))


def plot_model(model):
    if model._NAME == "Pln":
        ax = axes[0]
    else:
        ax = axes[1]
    absc = np.arange(0, len(model.running_times))
    sns.lineplot(
        x=absc,
        y=-np.array(model._elbos_list),
        color=my_palette[model.optim_choice],
        ax=ax,
        label=name_optim[model.optim_choice],
        linewidth=3,
    )


for optim in list_optim:
    model_pca = PlnPCA(data["endog"], optim_choice=optim)
    model_pca.fit(lr=0.01, nb_max_epoch=max_iter, only_iter=True)
    model_pln = Pln(data["endog"], optim_choice=optim)
    model_pln.fit(lr=0.01, nb_max_epoch=max_iter, only_iter=True)
    plot_model(model_pca)
    plot_model(model_pln)

handles, legend = axes[0].get_legend_handles_labels()
axbox = axes[1].get_position()

fig.legend(
    handles,
    legend,
    ncol=len(list_optim),
    loc="lower center",
    fontsize=18,
    bbox_to_anchor=[0, axbox.y0 - 0.18, 1, 1],
    bbox_transform=fig.transFigure,
)

axes[0].legend().remove()
axes[1].legend().remove()
axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Iteration number", fontsize=18)
axes[1].set_ylabel("Negative ELBO", fontsize=18)
axes[0].set_ylabel("Negative ELBO", fontsize=18)
# axes[1].legend()
axes[0].set_title("uPLN", fontsize=20)
axes[1].set_title("PLN-PCA", fontsize=20)

axes[0].tick_params(axis="both", which="major", labelsize=18)
axes[0].tick_params(axis="both", which="minor", labelsize=18)
axes[1].tick_params(axis="both", which="major", labelsize=18)
axes[1].tick_params(axis="both", which="minor", labelsize=18)


plt.savefig(
    "/home/bastien/These/manuscript/tex/figures/intro/optimizers.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
