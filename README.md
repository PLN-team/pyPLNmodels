![PyPI](https://img.shields.io/pypi/v/pyPLNmodels)
![GitHub](https://img.shields.io/github/license/PLN-team/pyPLNmodels)
![Python Versions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Pylint](https://img.shields.io/badge/pylint-checked-brightgreen)
![GPU Support](https://img.shields.io/badge/GPU-Supported-brightgreen)
[![codecov](https://codecov.io/gh/PLN-team/pyPLNmodels/graph/badge.svg?token=TNFROMLF9Z)](https://codecov.io/gh/PLN-team/pyPLNmodels)
[![last commit](https://img.shields.io/github/last-commit/PLN-team/pyPLNmodels)](https://github.com/PLN-team/pyPLNmodels)

# PLNmodels: Poisson lognormal models

> The Poisson lognormal model and its variants are used for the
> analysis of multivariate count data.
> This package implements efficient algorithms to extract
> meaningful insights from complex and difficult-to-interpret
> multivariate count data. It is designed to scale on large
> datasets, although it has memory limitations. Possible fields of
> application include:
> - Genomics/transcriptomics (e.g., the number of times a gene is expressed in a cell)
> - Ecology (e.g., species abundances)
> - Metagenomics (e.g., the number of Amplicon Sequence Variants detected in a sample)
> - Genetics (e.g., the number of crossovers (DNA exchanges) along a chromosome)
>
> One of the main functionalities of this package is to normalize
> count data to obtain more valuable insights. It also analyzes
> the significance of each variable, their correlations, and the
> weight of covariates (if available).

The package documentation can be found [here](https://pln-team.github.io/pyPLNmodels/).

## Getting started
[A notebook to get started can be found here](https://github.com/PLN-team/pyPLNmodels/blob/main/Getting_started.ipynb). A more in-depth tutorial is available [here](??).
If you need just a quick view of the package, see the quickstart next. Note
that an `R` version of the package is available [here](https://pln-team.github.io/PLNmodels/).

## 🛠 Installation

**pyPLNmodels** is available on [pypi](https://pypi.org/project/pyPLNmodels/).

### Package installation
```sh
pip install pyPLNmodels
```

## Statistical description

For those unfamiliar with Poisson or Gaussian random variables,
it's not necessary to delve into these statistical concepts. The
key takeaway is that this package analyzes multi-dimensional count
data, extracting significant information such as the mean,
relationships with covariates, and correlations between count
variables, in a manner appropriate for count data.

Consider $\mathbf Y$ a count matrix (denoted as `endog` in the package) consisting of $n$ rows and $p$ columns.
It is assumed that each individual $\mathbf Y_i$, that is the $i^{\text{th}}$
row of $\mathbf Y$, is independent from the others and follows a Poisson
lognormal distribution:

$$\mathbf Y_{i}\sim \mathcal P(\exp(\mathbf Z_{i})), \quad \mathbf Z_i \sim
\mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \mathbf \Sigma), (\text{PLN-equation})$$

where $\mathbf x_i \in \mathbb R^d$ (`exog`) and $\mathbf o_i \in \mathbb R^p$ (`offsets`) are
user-specified covariates and offsets. The matrix $\mathbf B$ is a $d\times p$
matrix of regression coefficients and $\mathbf \Sigma$ is a $p\times p$
covariance matrix. The goal is to estimate the parameters $\mathbf B$ and
$\mathbf \Sigma$, denoted as `coef` and `covariance` in the package,
respectively.

The PLN model described in the PLN-equation is the building block of many
different statistical tasks adequate for count data, by modifying the $Z_i$ latent variables. The package implements:

- Covariance analysis (`Pln`)
- Dimension reduction (`PlnPCA` and `PlnPCACollection`)
- Zero-inflation (`ZIPln`)
- Autoregressive models (`PlnAR`)
- Supervised clustering (`PlnLDA`)
- Unsupervised clustering (`PlnMixture`)
- Network inference (`PlnNetwork`)
- Zero-inflation and dimension reduction (`ZIPlnPCA`)
- Variance estimation (`PlnDiag`)

A normalization procedure adequate to count data can be applied
by extracting the `latent_variables` $\mathbf Z_i$ once the parameters are learned.

## ⚡️ Quickstart

The package comes with a single-cell RNA sequencing dataset to present the functionalities:
```python
from pyPLNmodels import load_scrna
data = load_scrna()
```

This dataset contains the number of occurrences of each gene in each cell in
`data["endog"]`. Each cell is labelled by its cell-type in `data["labels"]`.

### How to specify a model
Each model can be specified in two distinct manners:

* by formula (similar to R), where a data frame is passed and the formula is specified using the `from_formula` initialization:
```python
from pyPLNmodels import Pln
pln = Pln.from_formula("endog ~ 1  + labels ", data = data)
```

We rely on the [patsy](https://github.com/pydata/patsy) package for the formula parsing.

* by specifying the `endog`, `exog`, and `offsets` matrices directly:
```python
import numpy as np
endog = data["endog"]
exog = data["labels"]
offsets = np.zeros((endog.shape))
pln = Pln(endog=endog, exog=exog, offsets=offsets)
```

The parameters `exog` and `offsets` are optional. By default,
`exog` is set to represent an intercept, which is a vector of ones. Similarly,
`offsets` defaults to a matrix of zeros. The `offsets` should be on the scale of the log of the counts.

### Motivation

The count data is often very noisy, and inferring the latent variables $Z_i$
may reduce noise and increase signal. Suppose we try to infer the cell type of
each cell, using Linear Discriminant Analysis (LDA):
```python
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_classif_error(data, y):
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    lda = LDA()
    lda.fit(data_train, y_train)
    y_pred = lda.predict(data_test)
    return np.mean(y_pred != y_test)
```

Here is the classification error of the raw counts:
```python
data = load_scrna(n_samples=1000)
get_classif_error(data["endog"], data["labels"])
```
Output:
```
0.31
```

And here is the classification error of the latent variables $Z_i$:
```python
get_classif_error(Pln(data["endog"]).fit().latent_variables, data["labels"])
```
Output:
```
0.17
```

### Covariance analysis with the Poisson lognormal model (aka `Pln`)

This is the building-block of the models implemented in this package. It fits a Poisson lognormal model to the data:
```python
pln = Pln.from_formula("endog ~ 1  + labels ", data = data)
pln.fit()
print(pln)
transformed_data = pln.transform()
pln.show()
```

### Dimension reduction with the PLN Principal Component Analysis (aka `PlnPCA` and `PlnPCACollection`)

This model excels in dimension reduction and is capable of scaling to
high-dimensional count data ($p >> 1$), by constraining the covariance matrix
$\Sigma$ to be of low rank (the larger the rank, the slower the model but the
better the approximation). The user may specify the rank when creating the
`PlnPCA` object:
```python
from pyPLNmodels import PlnPCA
pca = PlnPCA.from_formula("endog ~ 1  + labels ", data = data, rank = 3).fit()
```

Multiple ranks can be simultaneously tested
within a single object (`PlnPCACollection`), and select the optimal model.
```python
from pyPLNmodels import PlnPCACollection
pca_col = PlnPCACollection.from_formula("endog ~ 1  + labels ", data = data, ranks = [3,4,5])
pca_col.fit()
print(pca_col)
pca_col.show()
best_pca = pca_col.best_model()
print(best_pca)
```

### Zero inflation with the Zero-Inflated PLN Model (aka `ZIPln` and `ZIPlnPCA`)

The `ZIPln` model, a variant of the PLN model, is designed to handle zero
inflation in the data. It is defined as follows:
$$Y_{ij}\sim \mathcal W_{ij} \times  P(\exp(Z_{ij})), \quad \mathbf Z_i \sim \mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \mathbf \Sigma), \quad W_{ij} \sim \mathcal B(\sigma( \mathbf x_i^{0^{\top}}\mathbf B^0_j))$$

This model is particularly beneficial when the data contains a significant
number of zeros. It incorporates additional covariates for the zero inflation
coefficient, which are specified following the pipe `|` symbol in the formula
or via the `exog_inflation` keyword. If not specified, it is set to the
covariates for the Poisson part.
```python
from pyPLNmodels import ZIPln
zi = ZIPln.from_formula("endog ~ 1 | 1 + labels", data = data).fit()
print(zi)
print("Transformed data shape: ", zi.transform().shape)
z_latent_variables = zi.transform()
w_latent_variables = zi.latent_prob
print(r'$Z$ latent variables shape', z_latent_variables.shape)
print(r'$W$ latent variables shape', w_latent_variables.shape)
```

Similar to the `PlnPCA` model, the `ZIPlnPCA` model is capable of dimension reduction.

### Network inference with the `PlnNetwork` model

The `PlnNetwork` model is designed to infer the network structure of the data.
It creates a network where the nodes are the count variables and the edges
represent the correlation between them. The sparsity of the network is ensured
via the `penalty` keyword. The larger the penalty, the sparser the network.
```python
from pyPLNmodels import PlnNetwork
net = PlnNetwork.from_formula("endog ~ 1  + labels ", data = data, penalty = 200).fit()
net.viz_network()
print(net.network)
```

### Supervised clustering with the `PlnLDA` model

One can do supervised clustering using Linear Discriminant Analysis
designed for count data.
```python
from pyPLNmodels import PlnLDA, plot_confusion_matrix
endog_train, endog_test = data["endog"][:500], data["endog"][500:]
labels_train, labels_test = data["labels"][:500], data["labels"][500:]
lda = PlnLDA(endog_train, clusters=labels_train).fit()
pred_test = lda.predict_clusters(endog_test)
plot_confusion_matrix(pred_test, labels_test)
```

### Unsupervised clustering with the `PlnMixture` model

```python
from pyPLNmodels import PlnMixture
mixture = PlnMixture.from_formula("endog ~ 0 ", data = data, n_cluster=3).fit()
mixture.show()
clusters = mixture.clusters
plot_confusion_matrix(clusters, data["labels"])
```

### Autoregressive models with the `PlnAR` model

The `PlnAR` model is designed to handle time series data. It is a simple (one step) autoregressive model that can be used to predict the next time point.
(This assumes the endog variable is a time series, which is not the case in the example below)
```python
from pyPLNmodels import PlnAR
ar = PlnAR.from_formula("endog ~ 1  + labels ", data = data).fit()
ar.show()
```

### Visualization

The package is equipped with a set of visualization functions designed to help
the user interpret the data. The `viz` function conducts `PCA`
on the latent variables. The `remove_exog_effect` keyword
removes the covariates' effect specified in the model when set to `True`.
Much more functionalities, depending on the model, are available. One can see the full list of available functions in the documentation and by printing the model:

```python
print(pln)
print(pca)
print(pca_col)
print(zi)
print(net)
print(lda)
print(mixture)
print(ar)
```

## 👐 Contributing

Feel free to contribute, but read the [CONTRIBUTING.md](https://github.com/PLN-team/pyPLNmodels/blob/main/CONTRIBUTING.md) first. A public roadmap will be available soon.

## ⚡️ Citations

Please cite our work using the following references:

- B. Batardiere, J.Kwon, J.Chiquet: *pyPLNmodels: A Python package to analyze
  multivariate high-dimensional count data.*
  [pdf](https://joss.theoj.org/papers/10.21105/joss.06969)

- J. Chiquet, M. Mariadassou and S. Robin: *Variational inference for
  probabilistic Poisson PCA, the Annals of Applied Statistics, 12:
  2674–2698, 2018.* [pdf](http://dx.doi.org/10.1214/18%2DAOAS1177)

- B. Batardiere, J.Chiquet, M.Mariadassou: *Zero-inflation in the Multivariate
  Poisson Lognormal Family.* [pdf](https://arxiv.org/abs/2405.14711)

- B. Batardiere, J.Chiquet, M.Mariadassou: *Evaluating Parameter Uncertainty in the Poisson
  Lognormal Model with Corrected Variational Estimators.* [pdf](https://arxiv.org/abs/2411.08524)

- J. Chiquet, M. Mariadassou, S. Robin: *The Poisson-Lognormal Model as a Versatile Framework for the Joint Analysis of Species Abundances.* [pdf](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2021.588292/full)

- J. Chiquet, S. Robin, M. Mariadassou: *Variational Inference for sparse network reconstruction from count data* [pdf](https://proceedings.mlr.press/v97/chiquet19a.html)
