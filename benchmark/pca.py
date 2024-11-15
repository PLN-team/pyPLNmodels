from pyPLNmodels import Pln, PlnPCA
import matplotlib.pyplot as plt
from benchmark import get_sc_mark_data
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import time

# tol = 1
tol = 0.0001
n = 1000
# p = 1000
p = 15000
Y, _, GT = get_sc_mark_data(max_n=n, dim=p)
viridis = sns.color_palette("viridis", n_colors=15)
print("length")
print(len(viridis))
colors = [
    viridis[0],
    viridis[0],
    viridis[0],
    viridis[4],
    viridis[4],
    viridis[9],
    viridis[9],
    viridis[14],
    viridis[14],
]

markers = ["o", "s", "v", "o", "s", "v", "o", "s", "v"]
dict_markers = dict(zip(np.unique(GT), markers))
dict_colors = dict(zip(np.unique(GT), colors))


plnpca = PlnPCA(Y)
t = time.time()
plnpca.fit(tol=tol, verbose=True)
print("time took pln:", (time.time() - t) / 60)

# plnpca.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 15), layout="constrained")
plnpca.viz(colors=GT, ax=axes[0], markers=dict_markers, dict_colors=dict_colors)


t = time.time()
pca = PCA(n_components=2)
y_pca = pca.fit_transform(np.log(Y + (Y == 0)))
print("time took pca", time.time() - t)
x = y_pca[:, 0]
y = y_pca[:, 1]
sns.scatterplot(
    x=x, y=y, ax=axes[1], hue=GT, markers=markers, palette=dict_colors, style=GT, s=150
)


fontsize = 28
axes[0].set_title("Principal Component Analysis with PLN-PCA", fontsize=fontsize + 2)
axes[0].set_xlabel("PC1", fontsize=fontsize)
axes[0].tick_params(axis="both", labelsize=fontsize - 2)
axes[0].set_ylabel("PC2", fontsize=fontsize)
axes[1].set_xlabel("PC1", fontsize=fontsize)
axes[1].tick_params(axis="both", labelsize=fontsize - 2)
axes[1].set_ylabel("PC2", fontsize=fontsize)
axes[1].set_title("Standard Principal Component Analysis", fontsize=fontsize + 2)
axes[0].get_legend().set_visible(False)

handles, legend = axes[1].get_legend_handles_labels()
axbox = axes[0].get_position()

fig.legend(
    handles,
    legend,
    ncol=3,
    loc="lower center",
    markerscale=3,
    fontsize=fontsize,
    bbox_to_anchor=[0, axbox.y0 - 0.28, 1, 1],
    bbox_transform=fig.transFigure,
)

axes[1].get_legend().set_visible(False)

plt.savefig(
    "../paper/figures/plnpca_vs_pca_last.png", format="png", bbox_inches="tight"
)
plt.show()
