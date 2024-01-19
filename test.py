from pyPLNmodels import get_real_count_data, ZIPln, Pln, get_simulated_count_data
from pyPLNmodels.models import Brute_ZIPln
import matplotlib.pyplot as plt
import seaborn as sns
import torch

endog, exog, offsets, cov, coef, coef_inflation = get_simulated_count_data(
    mean_XB=0.8,
    mean_infla=0.2,
    zero_inflated=True,
    return_true_param=True,
    nb_cov=0,
    add_const=True,
)

is_closed = True

brute_zi = Brute_ZIPln(
    endog, exog=exog, offsets=offsets, use_closed_form_prob=is_closed
)
brute_zi.fit()
print("brute elbo", brute_zi.elbo)
# brute_zi.plot_expected_vs_true()
print("reconstruction brute", brute_zi.reconstruction_error)


zi = ZIPln(endog, exog=exog, offsets=offsets, use_closed_form_prob=is_closed)
zi.fit()
print("reconstruction enhanced", zi.reconstruction_error)
print("elbo zi", zi.elbo)
# zi.plot_expected_vs_true()


def MSE(t):
    return torch.mean(t**2)


def MAE(t):
    return torch.mean(torch.abs(t))


# print('true coef infla', coef_inflation)
# print('zi coef infla', zi.coef_inflation)
# print('brute zi coef infla', brute_zi.coef_inflation)

print("Standard mse cov", MSE(brute_zi.covariance - cov))
print("Standard mse coef", MSE(brute_zi.coef - coef))
print("Standard mse coef_infla", MSE(brute_zi.coef_inflation - coef_inflation))
print(
    "Standard MAE proba_infla",
    MAE(torch.sigmoid(brute_zi.coef_inflation) - torch.sigmoid(coef_inflation)),
)

print("Enhanced mse cov", MSE(zi.covariance - cov))
print("Enhanced mse coef", MSE(zi.coef - coef))
print("Enhanced mse coef_infla", MSE(zi.coef_inflation - coef_inflation))
print(
    "Enhanced MAE proba_infla",
    MAE(torch.sigmoid(zi.coef_inflation) - torch.sigmoid(coef_inflation)),
)

fig, axes = plt.subplots(4, figsize=(20, 20))
sns.heatmap(torch.sigmoid(zi.coef_inflation), ax=axes[0])
sns.heatmap(torch.sigmoid(coef_inflation), ax=axes[1])
sns.heatmap(torch.sigmoid(brute_zi.coef_inflation), ax=axes[2])
sns.heatmap(coef > 1, ax=axes[3])
axes[0].set_title(r"Enhanced $\pi$")
axes[1].set_title(r"True $\pi$")
axes[2].set_title(r"Standard $\pi$")
axes[3].set_title("Threshold (XB > 1)")
plt.savefig("diff_pi.pdf", format="pdf")
plt.show()

fig, axes = plt.subplots(1, 3)
sns.heatmap(zi.covariance, ax=axes[0])
sns.heatmap(cov, ax=axes[1])
sns.heatmap(brute_zi.covariance, ax=axes[2])
axes[0].set_title("Enhanced")
axes[1].set_title("True")
axes[2].set_title("Standard")
plt.show()
