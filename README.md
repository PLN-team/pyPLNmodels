
# PLNmodels: Poisson lognormal models

> The Poisson lognormal model and variants can be used for analysis of mutivariate count data.
> This package implements
> efficient algorithms extracting meaningful data from difficult to interpret
> and complex multivariate count data. It has been built to scale on large datasets even
> though it has memory limitations. Possible fields of applications include
> - Genomics (number of times a gene is expressed in a cell)
> - Ecology (species abundances)
>
> One main functionality is to normalize the count data to obtain more valuable
> data. It also analyse the significance of each variable and their correlation as well as the weight of
> covariates (if available).
<!-- accompanied with a set of -->
<!-- > functions for visualization and diagnostic. See [this deck of -->
<!-- > slides](https://pln-team.github.io/slideshow/) for a -->
<!-- > comprehensive introduction. -->

##  Getting started
The getting started can be found [here](Getting_started.ipynb). If you need just a quick view of the package, see the quickstart next.

## 🛠 Installation

**pyPLNmodels** is available on
[pypi](https://pypi.org/project/pyPLNmodels/). The development
version is available on [GitHub](https://github.com/PLN-team/pyPLNmodels).

### Package installation

```
pip install pyPLNmodels
```

## Statistical description
Consider $\mathbf Y$ a count matrix consisting of $n$ rows and $p$ columns (denoted as ```endog``` in the package).
It is assumed that each individual $\mathbf Y_i$, that is the $i^{\text{th}}$
row of $\mathbf Y$, is independent from the others and follows a Poisson
lognormal distribution:
$$\mathbf Y_{i}\sim \mathcal P(\exp(\mathbf Z_{i})), \quad \mathbf \Z_i \sim
\mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \mathbf \Sigma),$$
where $\mathbf x_i \in \mathbb R^d$ (`exog`) and $\mathbf o_i \in \mathbb R^p$ (`offsets`) are
user-specified covariates and offsets. The matrix $\mathbf B$ is a $d\times p$
matrix of regression coefficients and $\mathbf \Sigma$ is a $p\times p$
covariance matrix. The goal is to estimate the parameters $\mathbf B$ and
$\mathbf \Sigma$, denoted as ```coef``` and ```covariance``` in the package,
respectively. A normalization procedure adequate to count data can be applied
by extracting the ```latent_variables``` $\mathbf Z_i$ once the parameters are learned.




## ⚡️ Quickstart

The package comes with an ecological data set to present the functionality:
```
import pyPLNmodels
from pyPLNmodels.models import PlnPCAcollection, Pln, ZIPln
from pyPLNmodels.oaks import load_oaks
oaks = load_oaks()
```

### How to specify a model
Each model can be specified in two distinct manners:
- by formula (similar to R), where a data frame is passed and the formula is specified using the  ```from_formula``` initialization:
```model = Model.from_formula("endog ~ 1  + covariate_name ", data = oaks)```
We rely to the [patsy](https://github.com/pydata/patsy) package for the formula parsing.
- by specifying the endog, exog, and offsets matrices directly:
```model = Model(endog = oaks["endog"], exog = oaks[["covariate_name"]], offsets = oaks[["offset_name"]])```

The parameters `exog` and `offsets` are optional. By default,
`exog` is set to represent an intercept, which is a vector of ones. Similarly,
`offsets` defaults to a matrix of zeros.


### Unpenalized Poisson lognormal model (aka PLN)

This is the building-block of the package. It fits a Poisson lognormal model to the data.
```
pln = Pln.from_formula("endog ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True)
pln.fit()
print(pln)
transformed_data = pln.transform()
pln.show()
```


### Rank Constrained Poisson lognormal for Poisson Principal Component Analysis (aka PLNPCA)

This model is efficient for dimension reduction and scales to high-dimensional
count data ($p >> 1$). It is a variant of the PLN model with a rank constraint on the
covariance matrix. It can be seen as a generalization of the [probabilistic
PCA](https://academic.oup.com/jrsssb/article/61/3/611/7083217) to count data,
with the rank giving the number of components of the probabilistic PCA.
The user can provide the rank of the covariance
matrix, with the additional capability to specify multiple ranks concurrently
in a single object, and retrieve the best model according to the AIC (default) or BIC criterion::
```
pca_col =  PlnPCAcollection.from_formula("endog ~ 1  + tree + dist2ground + orientation ", data = oaks, take_log_offsets = True, ranks = [3,4,5])
pca_col.fit()
print(pca_col)
pca_col.show()
best_model = pca_col.best_model()
best_model.show()
transformed_data = best_model.transform(project = True)
print('Original data shape: ', oaks["endog"].shape)
print('Transformed data shape: ', transformed_data.shape)
```

A correlation circle can be plotted to visualize the correlation between the variables and the components:
```
best_model.plot_pca_correlation_circle(["var_1","var_2"], indices_of_variables = [0,1])
```


### Zero inflated Poisson Log normal Model (aka ZIPln)

The ```ZiPln``` model is a variant of the PLN model that accounts for zero
inflation in the data:
$$Y_{ij}\sim \mathcal W_{ij} \times  P(\exp(Z_{ij})), \quad \mathbf \Z_i \sim
\mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \mathbf \Sigma), W_{ij} \sim \mathcal B(\sigma( \mathbf x_i^{0^{\top}}\mathbf B^0_j))$$
It is particularly useful when the data contains many
zeros. The model accounts for additional covariates for the zero inflation coefficient, and are specified using the pipe `|` symbol in the formula:`
```
zi =  ZIPln.from_formula("endog ~ 1  + tree + dist2ground + orientation | 1 + tree", data = oaks, take_log_offsets = True)
zi.fit()
print(zi)
z_latent_variables, w_latent_variables = zi.transform(return_latent_prob = True)
print(r'$Z$ latent variables shape', z_latent_variables.shape)
print(r'$W$ latent variables shape', w_latent_variables.shape)
```
The transformed data is composed of both the latent

### Visualization

The package comes with a set of visualization functions to help the user
interpret the data.
```
best_model.viz()
```



## 👐 Contributing

Feel free to contribute, but read the [CONTRIBUTING.md](https://forgemia.inra.fr/bbatardiere/pyplnmodels/-/blob/main/CONTRIBUTING.md) first. A public roadmap will be available soon.

## ⚡️ Citations

Please cite our work using the following references:
-   J. Chiquet, M. Mariadassou and S. Robin: Variational inference for
    probabilistic Poisson PCA, the Annals of Applied Statistics, 12:
        2674–2698, 2018. [link](http://dx.doi.org/10.1214/18%2DAOAS1177)
