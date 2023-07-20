from pyPLNmodels import Pln, PlnPCA, PlnPCAcollection
from pyPLNmodels import get_real_count_data, get_simulated_count_data
import torch

# counts = get_real_count_data()

counts, covariates, offsets = get_simulated_count_data(n_samples=20000)
# pca = Pln(counts,exog = covariates, offsets = offsets, batch_size = 20)
# pca = PlnPCA(counts, batch_size = None, rank=15)
# pca.fit(nb_max_iteration=8000, class_optimizer=torch.optim.Rprop)
pca = PlnPCA(counts, batch_size=None, rank=15)
pca.fit(nb_max_iteration=20, class_optimizer=torch.optim.Rprop, verbose=True)
# pca = Pln(counts, batch_size = None)
# pca.fit(nb_max_iteration=1000, class_optimizer=torch.optim.Rprop)
# pca = Pln(counts, batch_size = 20)
# pca.fit(nb_max_iteration=1000, class_optimizer=torch.optim.Adam)
print(pca)
pca.show()
