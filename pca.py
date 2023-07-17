from pyPLNmodels import Pln, PlnPCA
import matplotlib.pyplot as plt
from benchmark import get_sc_mark_data
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np


n = 10000
p = 80
Y, _, GT = get_sc_mark_data(max_n=n, dim=p)


plnpca = PlnPCA(Y, rank=8)
plnpca.fit(tol=0.05)

fig, axes = plt.subplots(2)

plnpca.viz(colors=GT, ax=axes[0])

pca = PCA(n_components=2)
y_pca = pca.fit_transform(np.log(Y + (Y == 0)))
sns.scatterplot(x=y_pca[:, 0], y=y_pca[:, 1], ax=axes[1], hue=GT)
axes[0].set_title("Principal Component Analysis with PlnPCA")
axes[1].set_title("Standard Principal Component Analysis")
plt.show()
