from pyPLNmodels import Pln, PlnPCA
import matplotlib.pyplot as plt
from benchmark import get_sc_mark_data
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import time


n = 1000
p = 15000
Y, _, GT = get_sc_mark_data(max_n=n, dim=p)


plnpca = PlnPCA(Y, rank=2)
t = time.time()
plnpca.fit(tol=0.001, verbose=True, nb_max_iteration=700)
print("time took pln:", (time.time() - t) / 60)
plnpca.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 10))


plnpca.viz(colors=GT, ax=axes[0])
t = time.time()
pca = PCA(n_components=2)
y_pca = pca.fit_transform(np.log(Y + (Y == 0)))
print("time took pca", time.time() - t)
sns.scatterplot(x=y_pca[:, 0], y=y_pca[:, 1], ax=axes[1], hue=GT)
axes[0].set_title("Principal Component Analysis with PlnPCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Standard Principal Component Analysis")
plt.savefig("paper/plnpca_vs_pca.png", format="png")
plt.show()
