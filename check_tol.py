from pyPLNmodels import Pln, PlnPCA, get_pln_simulated_count_data, load_scrna
import matplotlib.pyplot as plt
import numpy as np

n_samples = 200
nb_epoch = 15000
endog, exog, offsets = get_pln_simulated_count_data(n_samples=n_samples, dim=100)

data = load_scrna()


nb_batch = 3
thresholds = [1e-4, 1e-5, 1e-6, 1e-7]
batch_sizes = np.linspace(50, n_samples, nb_batch).astype(int)
fig, axes = plt.subplots(3)


colors_batch = np.linspace(150, 200, len(batch_sizes))
colors_batch = colors_batch.reshape(-1, 1)
colors_batch = np.repeat(colors_batch, 3, axis=1)
dict_colors_batch = {
    batch_sizes[i]: colors_batch[i] / 235 for i in range(len(batch_sizes))
}
colors_th = np.linspace(150, 200, len(thresholds))
colors_th = colors_th.reshape(-1, 1)
colors_th = np.repeat(colors_th, 3, axis=1)
dict_colors_th = {thresholds[i]: colors_th[i] / 235 for i in range(len(thresholds))}


firsts = {threshold: {"coef": [], "sigma": []} for threshold in thresholds}

for batch_size in batch_sizes:
    batch_size = int(batch_size)
    # pln = PlnPCA(endog, exog=exog, offsets=offsets, batch_size=batch_size)
    pln = Pln.from_formula("endog ~ 1", data=data, batch_size=batch_size)
    pln.fit(nb_max_iteration=int(nb_epoch * n_samples / batch_size), tol=0)
    pln._all_mses
    pln._display_norm(axes[0], label=batch_size, color=dict_colors_batch[batch_size])
    pln._diff_mses(ax=axes[1], label=batch_size, color=dict_colors_batch[batch_size])
    diff_mses = pln._get_diff_mses()
    for threshold in thresholds:
        inf = diff_mses < threshold
        argmax = np.argmax(inf, axis=1)
        firsts[threshold]["coef"].append(argmax[0])
        firsts[threshold]["sigma"].append(argmax[1])
for threshold in thresholds:
    axes[2].plot(
        batch_sizes,
        firsts[threshold]["coef"],
        label=str(threshold) + " coef",
        color=dict_colors_th[threshold],
    )
    axes[2].plot(
        batch_sizes,
        firsts[threshold]["sigma"],
        label=str(threshold) + " sigma",
        color=dict_colors_th[threshold],
    )
axes[2].legend()


axes[1].legend()
axes[0].legend()
plt.show()
