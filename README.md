
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
A notebook to get started can be found
[HERE](https://github.com/PLN-team/pyPLNmodels/blob/main/Getting_started.ipynb).
If you need just a quick view of the package, see the quickstart next.

## 🛠 Installation

**pyPLNmodels** is available on
[pypi](https://pypi.org/project/pyPLNmodels/). The development
version is available on [GitHub](https://github.com/PLN-team/pyPLNmodels) and [GitLab](https://gitlab.com/Bastien-mva/pyplnmodels).

### Package installation
```
pip install pyPLNmodels
```


## Statistical description

For those unfamiliar with the concepts of Poisson or Gaussian random variables,
it is not necessary to delve into these statistical descriptions. The key
takeaway is as follows:
This package is designed to analyze multi-dimensional count data. It
effectively extracts significant information, such as
the mean, the relationships with covariates, and the correlation between count
variables, in a manner appropriate for count data.

Consider $\mathbf Y$ a count matrix (denoted as ```endog``` in the package) consisting of $n$ rows and $p$ columns.
It is assumed that each individual $\mathbf Y_i$, that is the $i^{\text{th}}$
row of $\mathbf Y$, is independent from the others and follows a Poisson
lognormal distribution:
$$\mathbf Y_{i}\sim \mathcal P(\exp(\mathbf Z_{i})), \quad \mathbf Z_i \sim
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

* by formula (similar to R), where a data frame is passed and the formula is specified using the  ```from_formula``` initialization:

```model = Model.from_formula("endog ~ 1  + covariate_name ", data = oaks)# not run```

We rely to the [patsy](https://github.com/pydata/patsy) package for the formula parsing.

* by specifying the endog, exog, and offsets matrices directly:

```model = Model(endog = oaks["endog"], exog = oaks[["covariate_name"]], offsets = oaks[["offset_name"]])# not run```

The parameters `exog` and `offsets` are optional. By default,
`exog` is set to represent an intercept, which is a vector of ones. Similarly,
`offsets` defaults to a matrix of zeros.

### Unpenalized Poisson lognormal model (aka `Pln`)

This is the building-block of the models implemented in this package. It fits a Poisson lognormal model to the data:
```
pln = Pln.from_formula("endog ~ 1  + tree ", data = oaks)
pln.fit()
print(pln)
transformed_data = pln.transform()
pln.show()
```

### Rank Constrained Poisson lognormal for Poisson Principal Component Analysis (aka `PlnPCA` and `PlnPCAcollection`)

This model excels in dimension reduction and is capable of scaling to
high-dimensional count data ($p >> 1$). It represents a variant of the PLN
model, incorporating a rank constraint on the covariance matrix. This can be
interpreted as an extension of the [probabilistic
PCA](https://academic.oup.com/jrsssb/article/61/3/611/7083217) for
count data, where the rank determines the number of components in the
probabilistic PCA. Users have the flexibility to define the rank of the
covariance matrix via the `rank` keyword of the `PlnPCA` object. Furthermore, they can specify multiple ranks simultaneously
within a single object (`PlnPCAcollection`), and then select the optimal model based on either the
AIC (default) or BIC criterion:
```
pca_col =  PlnPCAcollection.from_formula("endog ~ 1  + tree ", data = oaks, ranks = [3,4,5])
pca_col.fit()
print(pca_col)
pca_col.show()
best_pca = pca_col.best_model()
best_pca.show()
transformed_data = best_pca.transform(project = True)
print('Original data shape: ', oaks["endog"].shape)
print('Transformed data shape: ', transformed_data.shape)
```

A correlation circle may be employed to graphically represent the relationship
between the variables and the components:
```
best_pca.plot_pca_correlation_circle(["var_1","var_2"], indices_of_variables = [0,1])
```


### Zero inflated Poisson Log normal Model (aka `ZIPln`)

The `ZiPln` model, a variant of the PLN model, is designed to handle zero
inflation in the data. It is defined as follows:

$$Y_{ij}\sim \mathcal W_{ij} \times  P(\exp(Z_{ij})), \quad \mathbf Z_i \sim \mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \mathbf \Sigma), \quad W_{ij} \sim \mathcal B(\sigma( \mathbf x_i^{0^{\top}}\mathbf B^0_j))$$

This model is particularly beneficial when the data contains a significant
number of zeros. It incorporates additional covariates for the zero inflation
coefficient, which are specified following the pipe `|` symbol in the formula or via the `exog_inflation` keyword. If not specified, it is set to the covariates for the Poisson part.

```
zi =  ZIPln.from_formula("endog ~ 1  + tree | 1 + tree", data = oaks)
zi.fit()
print(zi)
print("Transformed data shape: ", zi.transform().shape)
z_latent_variables, w_latent_variables = zi.transform(return_latent_prob = True)
print(r'$Z$ latent variables shape', z_latent_variables.shape)
print(r'$W$ latent variables shape', w_latent_variables.shape)
```

By default, the transformation of the data returns only the $\mathbf Z$ latent
variable. However, if the `return_latent_prob`
parameter is set to `True`, the transformed data will include both the latent
variables $\mathbf W$ and $\mathbf Z$. Here, $\mathbf W$ accounts for the zero
inflation, while $\mathbf Z$ accounts for the Poisson parameter.

### Visualization

The package is equipped with a set of visualization functions designed to
help the user interpret the data. The `viz` function conducts Principal
Component Analysis (PCA) on the latent variables, while the `viz_positions` function
carries out PCA on the latent variables, adjusted for covariates. Additionally,
the `viz_prob` function provides a visual representation of the zero-inflation
probability.

```
best_pca.viz(colors = oaks["tree"])
best_pca.viz_positions(colors = oaks["dist2ground"])
pln.viz(colors = oaks["tree"])
pln.viz_positions(colors = oaks["dist2ground"])
zi.viz(colors = oaks["tree"])
zi.viz_positions(colors = oaks["dist2ground"])
zi.viz_prob(colors = oaks["tree"])
```

## 👐 Contributing

Feel free to contribute, but read the [CONTRIBUTING.md](https://forgemia.inra.fr/bbatardiere/pyplnmodels/-/blob/main/CONTRIBUTING.md) first. A public roadmap will be available soon.

## ⚡️ Citations

Please cite our work using the following references:

-   J. Chiquet, M. Mariadassou and S. Robin: Variational inference for
    probabilistic Poisson PCA, the Annals of Applied Statistics, 12:
        2674–2698, 2018. [pdf](http://dx.doi.org/10.1214/18%2DAOAS1177)

-  B. Batardiere, J.Chiquet, M.Mariadassou: Zero-inflation in the Multivariate
   Poisson Lognormal Family. [pdf](https://arxiv.org/abs/2405.14711)
