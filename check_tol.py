from pyPLNmodels import Pln, PlnPCA, get_pln_simulated_count_data, load_scrna
import matplotlib.pyplot as plt
import numpy as np

n_samples = 200
nb_epoch = 200
endog, exog, offsets = get_pln_simulated_count_data(n_samples=n_samples, dim=100)

# data = load_scrna()


nb_batch = 2
batch_sizes = np.linspace(50, n_samples, nb_batch).astype(int)
fig, axes = plt.subplots(2)

for batch_size in batch_sizes:
    batch_size = int(batch_size)
    pln = PlnPCA(endog, exog=exog, offsets=offsets, batch_size=batch_size)
    # pln = Pln.from_formula("endog ~ 1", data=data, batch_size=batch_size)
    pln.fit(nb_max_iteration=int(nb_epoch * n_samples / batch_size), tol=0)
    pln._all_mses
    pln._display_norm(axes[0])
    pln._diff_mses(ax=axes[1], label=batch_size)

axes[1].legend()
plt.show()
