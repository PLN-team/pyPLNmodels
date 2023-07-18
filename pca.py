from pyPLNmodels import Pln, PlnPCA
import matplotlib.pyplot as plt
from benchmark import get_sc_mark_data
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

n = 1000
p = 4000
Y, _, GT = get_sc_mark_data(max_n=n, dim=p)


plnpca = PlnPCA(Y, rank=50)
plnpca.fit(tol=0.05)


plnpca.viz(colors=GT, ax=axes[0])

pca = PCA(n_components=2)
y_pca = pca.fit_transform(np.log(Y + (Y == 0)))
sns.scatterplot(x=y_pca[:, 0], y=y_pca[:, 1], ax=axes[1], hue=GT)
axes[0].set_title("Principal Component Analysis with PlnPCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Standard Principal Component Analysis")
plt.savefig("paper/plnpca_vs_pca.png", format="png")
plt.show()
