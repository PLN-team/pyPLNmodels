from pyPLNmodels import Pln, PlnPCA
import matplotlib.pyplot as plt
from benchmark import get_sc_mark_data
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import time


# n = 1000
# p = 15000
n = 300
p = 150
Y, _, GT = get_sc_mark_data(max_n=n, dim=p)


plnpca = PlnPCA(Y, rank=2)
t = time.time()
plnpca.fit(tol=0.001, verbose=True, nb_max_iteration=70)
print("time took pln:", (time.time() - t) / 60)
# plnpca.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
plnpca.viz(ax=axes[0])
t = time.time()
pca = PCA(n_components=2)
y_pca = pca.fit_transform(np.log(Y + (Y == 0)))
print("time took pca", time.time() - t)
x = y_pca[:, 0]
y = y_pca[:, 1]
sns.scatterplot(x=x, y=y, ax=axes[1], hue=GT)
axes[0].set_title("Principal Component Analysis with PlnPCA")
fontsize = 20
axes[0].set_xlabel("PC1", fontsize=fontsize)
axes[0].tick_params(axis="both", labelsize=fontsize - 5)
axes[0].set_ylabel("PC2", fontsize=fontsize)
axes[1].set_xlabel("PC1", fontsize=fontsize)
axes[1].tick_params(axis="both", labelsize=fontsize - 5)
axes[1].set_ylabel("PC2", fontsize=fontsize)
axes[1].set_title("Standard Principal Component Analysis")
plt.savefig("paper/figures/plnpca_vs_pca.png", format="png")
plt.show()
