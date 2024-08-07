---
title: 'pyPLNmodels: A Python package to analyze multivariate high-dimensional count data'
tags:
  - Python
  - count data
  - multivariate
  - genomics
  - ecology
  - high-dimension
authors:
  - name: Bastien Batardiere
    orcid: 0009-0001-3960-7120
    corresponding: true
    affiliation: 1
  - name: Joon Kwon
    affiliation: 1
  - name: Julien Chiquet
    affiliation: 1
affiliations:
 - name: Université Paris-Saclay, AgroParisTech, INRAE, UMR MIA Paris-Saclay
   index: 1
date: 28 June 2023
bibliography: paper.bib
---

# Summary
High dimensional count data are complex to analyze as is, and normalization must
be performed, but standard normalization does not fit the characteristics of
count data. The Poisson LogNormal(PLN)  [@PLN] and its Principal Component
Analysis variant PLN-PCA [@PLNPCA] are two-sided latent variable models allowing both
suitable normalization and analysis of multivariate count data, implemented in this package.

Consider $\mathbf Y$ a count matrix consisting of $n$ rows and $p$ columns. The
$\mathcal P$ (resp. $\mathcal N$) denotes a Poisson (resp. Normal)
distribution. It is assumed that each individual $\mathbf Y_i$, that is the $i^{\text{th}}$
row of $\mathbf Y$, is independent of the others and follows a Poisson
lognormal distribution:
$$\mathbf Y_{i}\sim \mathcal P(\exp(\mathbf Z_{i})), \quad \mathbf Z_i \sim
\mathcal N(\mathbf o_i + \mathbf B ^{\top} \mathbf x_i, \boldsymbol{\Sigma}),$$
where $\mathbf x_i \in \mathbb R^d$ and $\mathbf o_i \in \mathbb R^p$ are
user-specified covariates and offsets (with default values if not available). The matrix $\mathbf B$ is a $d\times p$
matrix of regression coefficients and $\boldsymbol{\Sigma}$ is a $p\times p$ covariance matrix. The variables $\mathbf Z_i$, known as *latent variables*,
are not directly observable. However, from a statistical perspective,
they provide more informative insights compared to the observed variables
$\mathbf Y_i$. The unknown parameters $\mathbf B$ and
$\boldsymbol{\Sigma}$ facilitates the analysis of
dependencies between variables and
the impact of covariates. The primary objective of the package is to estimate these
parameters and retrieve the latent variables $\mathbf Z_i$.  Extracting
those latent variables may serve as a normalization procedure adequate to count data.

The only difference between the PLN and PLN-PCA models is that the latter
assumes a low-rank structure on the covariance matrix, which is helpful for
dimension
reduction. Other variants of the PLN model exist, which are detailed in
the work of [@PLNmodels].

# Fields of applications and functionalities
Possible fields of applications include
\begin{itemize}
\item Ecology: Joint analysis of species abundances is a common task in
ecology, whose goal is to understand the interaction between species to
characterize a community, given a matrix of abundances in different sites with abundances given by
$$Y_{ij} = \text{number of species } j \text{ observed in site } i .$$
<!-- Specifically, it aims to establish potential dependencies, competitive interactions, and predatory dynamics. -->
Additionally, the PLN models seek to explain the impact of covariates (when available), such as temperature, altitude, and other
  relevant factors on the observed abundances.
\item Genomics: High throughput sequencing technologies now allow quantification, at the level of
individual cells, various measures from the genome of humans, animals, and plants. Single-cell Ribonucleic Acid
sequencing (scRNA-seq) is one of those and measures the expression of genes at the level of individual cells. For
cell $i$ and gene $j$, the counts $Y_{ij}$ is given by
$$Y_{ij} = \text{number of times gene } j \text{ is expressed in cell } i.$$
One of the challenges with scRNA-seq data is managing the high
dimensionality, necessitating dimension reduction techniques adequate to count data.
\end{itemize}
The PLN and PLN-PCA variants are implemented in the ```pyPLNmodels``` package
introduced here, whose main functionalities are
\begin{itemize}
\item Normalize count data to obtain more valuable data
\item Analyse the significance of each variable and their correlation
\item Perform regression when covariates are available
\item Reduce the number of features with PLN-PCA
\end{itemize}
The ```pyPLNmodels```[^pyplnmodels]  package has been designed to efficiently process
extensive datasets in a reasonable time and incorporates GPU
acceleration for better scalability.


[^pyplnmodels]: https://github.com/PLN-team/pyPLNmodels
[^plnmodels]: https://github.com/PLN-team/PLNmodels


To illustrate the primary model's interest, we display below a visualization of
the first two principal components when Principal
Component Analysis (PCA) is performed with the PLN-PCA model (left, ours) and standard PCA on
the log normalized data (right). The data considered is the `scMARK` benchmark [@scMark] described in the
benchmark section. We kept 1000 samples for illustration
purposes. The computational time for fitting PLN-PCA is 23 seconds (on GPU), whereas
standard PCA requires 0.7 second.

![PLN-PCA (left, ours) and standard PCA on log normalized data (right). Each cell is
identified by its respective cell type. This categorization is done solely to demonstrate the
method's ability to differentiate between various cell types. Unlike the
standard Principal Component Analysis (PCA), which fails to distinguish between
different cell types, the PLN-PCA method is capable of doing
so.](figures/plnpca_vs_pca_last.png)

# Statement of need
While the R-package ```PLNmodels``` [@PLNmodels] implements PLN models (including some variants), the Python package
```pyPLNmodels``` based on Pytorch [@Pytorch] has been built to handle
large datasets of count data, such as scRNA-seq data. Real-world scRNA-seq
datasets typically involve thousands of cells ($n \approx 20000$) with
thousands of genes ($\approx 20000$), resulting in a matrix of size $\approx
20000 \times 20000$.

The `statsmodels` [@statsmodels] Python package allows to deal with count data
thanks to the Generalized Linear Models `PoissonBayesMixedGLM` and
`BinomialBayesMixedGLM` classes. We stand out from this package by allowing covariance
between features and performing Principal Component Analysis adequate to count data.

The `GLLVM` package [@GLLVM] offers a broader scope of modeling
capabilities, enabling the incorporation of Poisson distribution as well as
Binomial or Negative Binomial distributions
and an additional zero-inflation component. However, its scalability is
notably inferior to our proposed methodology. Our approach, specifically
the PLN-PCA model, demonstrates superior scalability, effectively
accommodating datasets with tens of thousands of variables. In contrast, the
PLN model handles thousands of variables within a reasonable computational timeframe. In
contrast, ```GLLVM``` struggles to scale beyond a few hundred variables within
practical computational limits.


# Benchmark
We compare
\begin{itemize}
\item PLN and PLN-PCA variants fitted with  \verb|pyPLNmodels| on CPU: \textbf{py-PLN-CPU} and \textbf{py-PLN-PCA-CPU}
\item PLN and PLN-PCA variants fitted with  \verb|pyPLNmodels| on GPU: \textbf{py-PLN-GPU} and \textbf{py-PLN-PCA-GPU}
\item PLN and PLN-PCA variants fitted with  \verb|PLNmodels| (on CPU): \textbf{R-PLN} and \textbf{R-PLN-PCA}
\item \verb|GLLVM| (on CPU): \textbf{GLLVM}
\end{itemize}
on the `scMARK` dataset, a benchmark for scRNA data, with
$n=19998$ cells (samples) and 14059 genes (variables) are available.
We plot below the running times required to fit such models when the number of variables (i.e.
genes) grows when $n = 100,1000, 19998$. We used $q =5$ Principal Components when fitting each
PLN-PCA model and the number of latent variables LV=$2$ for the ```GLLVM``` model.
For each model, the fitting process was halted if the running time exceeded
10,000 seconds. We were unable to run ```GLLVM``` for $n = 19998$ due to CPU memory
limitations (64 GB RAM). Similarly, ```py-PLN-PCA-GPU``` could not be run when
$n=19998$ and $p\geq13000$ as it exceeded the GPU memory capacity (24 GB RAM).


![Running time analysis on the scMARK benchmark.](figures/plots_benchmark.pdf)
Each package uses variational inference, maximizing an Evidence
Lower BOund(ELBO) approximating the log-likelihood of the model.
```GLLVM``` uses an alternate-optimization scheme, fitting alternatively a
Negative Binomial (NB) Generalized Linear Model(GLM) and two penalized NB GLM
coupled with a fixed-point algorithm, while ```pyPLNmodels``` and
```PLNmodels``` uses gradient ascent to maximize the ELBO.
```PLNmodels``` uses C++ backend along with ```nlopt```[@nlopt] optimization library.
The backend of ```GLLVM``` is implemented in C++, while ```pyPLNmodels``` leverages the
automatic differentiation from Pytorch to compute the gradients of the ELBO. Each
PLN-PCA model is estimated using comparable variational inference methods.
However, the variational approximation for the PLN model in the
```pyPLNmodels``` version is more efficient than its counterpart in
```PLNmodels```.

# Ongoing work
A zero-inflated version of the PLN model is currently under development, with a
preprint [@PLNzero] expected to be published shortly.

# Acknowledgements
The authors would like to thank Jean-Benoist Léger for the time spent giving
precious advice on how to build a proper Python package.

# Fundings
Bastien Bartardière and Julien Chiquet are supported by
the French ANR grant ANR-18-CE45-0023 Statistics and Machine Learning for Single Cell Genomics (SingleStatOmics).

# References
