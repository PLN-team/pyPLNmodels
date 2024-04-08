from pyPLNmodels import ZIPln, Pln
from pyPLNmodels.models import Brute_ZIPln
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {}
data["endog"] = pd.read_csv("reduced/endog.csv").drop(columns="Unnamed: 0")
p = data["endog"].shape[1]
data["site"] = pd.read_csv("reduced/site.csv").drop(columns="Unnamed: 0").squeeze()
data["time"] = pd.read_csv("reduced/time.csv").drop(columns="Unnamed: 0").squeeze()


best = (data["endog"] > 0).mean(axis=0) > 0.05
data["endog"] = data["endog"].loc[:, best]

data["offsets"] = np.log(
    pd.read_csv("reduced/offsets.csv")
    .drop(columns="Unnamed: 0")
    .values.repeat(p, axis=1)
)[:, best]


pln = Brute_ZIPln.from_formula(
    "endog ~ 1 + site+time",
    data=data,
    zero_inflation_formula="global",
    use_closed_form_prob=False,
)
pln.fit(verbose=True, tol=0, nb_max_iteration=100)
pca = PCA(n_components=2)
fig, axes = plt.subplots(2, 1)
y = pln.latent_mean
y = pca.fit_transform(y)
y_moinsXB = pln.latent_mean - pln.mean_gaussian
y_moinsXB = pca.fit_transform(y_moinsXB)
data["site"] = data["site"].iloc[(~pln._samples_only_zeros).numpy(),]
data["time"] = data["time"].iloc[(~pln._samples_only_zeros).numpy(),]
# df = pd.DataFrame([{"x": np.array(y[:,0]), "y": np.array(y[:,1]),"color" : data["site"].values}])
df = pd.DataFrame(
    {"x": np.array(y[:, 0]), "z": np.array(y[:, 1]), "color": data["site"]}
)
df_moins_XB = pd.DataFrame(
    {
        "x": np.array(y_moinsXB[:, 0]),
        "z": np.array(y_moinsXB[:, 1]),
        "color": data["site"],
    }
)
# sns.scatterplot(y, hue = data["site"].values)
print("data", df["color"])
print("df ", df.shape)

sns.scatterplot(df, x="x", y="z", hue="color", ax=axes[0], style=data["time"])
sns.scatterplot(df_moins_XB, x="x", y="z", hue="color", ax=axes[1], style=data["time"])
plt.show()

print("loglike", pln.loglike)
# print('p:', pln.dim)
# print('nb param', pln.number_of_parameters)
# pln.show()
