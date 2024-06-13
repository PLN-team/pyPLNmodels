---
title: 'pyPLNmodels: A Python package to analyse multivariate high-dimensional count data'
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
    corresponding: true # (This is how to denote the corresponding author)
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
High dimensional count data are hard to analyse as is, and normalization must
be performed but standard normalization does not fit to the characteristics of
count data. The Poisson LogNormal(PLN)  [@PLN] and its Principal Component
Analysis counterpart PLN-PCA [@PLNPCA] are two-sided models allowing both
suitable normalization and analysis of multivariate count data. It establishes
 patterns and dependency structures through a correlation matrix. Each model is
implemented in the `pyPLNmodels` package introduced here.
Possible fields of applications include
<!-- Multivariate abundance data, consisting of observations of multiple interacting -->
<!-- species from a set of samples, are often collected in ecological studies to -->
<!-- characterize a community or assemblage of organisms. -->

\begin{itemize}
\item Ecology: Joint analysis of species abundances is a common task in
ecology, whose goal is to understand the interaction between species to
characterize a community, given a matrix of abundances in different sites with abundances given by
$$Y_{ij} = \text{number of species } j \text{ observed in site } i .$$
Specifically, it aims to establish potential dependencies, competitive interactions, and predatory dynamics.
Additionally, the PLN models seek to explain the impact of covariates (when available), such as temperature, altitude, and other
  relevant factors, on the observed abundances.
\item Genomics: High
throughput sequencing technologies now allow to quantify at the level of
individual cells various measures from the genome (DNA, Ribonucleic Acid(RNA),
chromatin and proteins) of humans, animals and plants. Single-cell RNA
sequencing (scRNA-seq) is one of those and measure the expression of genes at the level of individual cells. For
cell $i$ and gene $j$, the counts $Y_{ij}$ is given by
$$Y_{ij} = \text{number of times gene } j \text{ is expressed in cell } i.$$
One of the challenges with scRNA-seq data is managing the high
dimensionality, necessitating variable reduction techniques adequate to count data.
\end{itemize}
The models can deal with offsets when needed. The main functionalities of the `pyPLNmodels` package are
\begin{itemize}
\item Normalize count data to obtain more valuable data
\item Analyse the significance of each variable and their correlation
\item Perform regression when covariates are available
\item Reduce the number of features with PLN-PCA
\item Visualize the normalized data
\end{itemize}
The PLN model is versatile and can be extended beyond PCA to accommodate other
multivariate statistical tasks. These tasks include Clustering, Linear
Discriminant Analysis, and Network inference, implemented (only) in the R-package ```PLNmodels```[^plnmodels] and described in [@PLNmodels].

The ```pyPLNmodels```[^pyplnmodels]  package has been designed to efficiently process
extensive datasets in a reasonable time and incorporates GPU
acceleration.


[^pyplnmodels]: https://github.com/PLN-team/pyPLNmodels
[^plnmodels]: https://github.com/PLN-team/PLNmodels


To illustrate the main model's interest, we display below a visualization of the first two principal components when Principal
Component Analysis (PCA) is performed with the PLN-PCA model (left) and standard PCA on
the log normalized data (right). The data considered is the `scMARK` benchmark [@scMark] described in the
benchmark section. We kept 1000 samples for illustration
purposes. The computational time for fitting PLN-PCA is 23 seconds (on GPU), whereas
standard PCA requires 0.7 second.

![PLN-PCA (left) and standard PCA on log normalized data (right). The computational time for fitting PLN-PCA is 23 seconds (on GPU), and 0.7 second for
standard PCA.](figures/plnpca_vs_pca.png)

# Statement of need
While the R-package `PLNmodels` [@PLNmodels] implements PLN models, the python package
`pyPLNmodels` based on Pytorch [@Pytorch] has been built to handle
large datasets of count data, such as scRNA-seq data. Real-world scRNA-seq datasets typically involves thousands of cells ($n \approx 20000$) with
thousands of genes ($\approx 20000$), resulting in a matrix of size $\approx
20000 \times 20000$. The package has GPU support for a better scalability.

The `statsmodels` [@statsmodels] python package allows to deal with count data
thanks to the Generalized Linear Models `PoissonBayesMixedGLM` and
`BinomialBayesMixedGLM` classes. We stand out from this package by allowing covariance
between features and performing Principal Component Analysis adequate to count data.

The `gllvm` package [@GLLVM] offers a broader scope of modeling
capabilities, enabling the incorporation of not
only Poisson distribution but also Binomial or negative Binomial distributions,
along with an additional zero-inflation component. However, its scalability is
notably inferior to our proposed methodology. Our approach, specifically
the PLN-PCA model, demonstrates superior scalability, effectively
accommodating datasets with tens of thousands of variables, while the PLN model
handles thousands of variables within a reasonable computational timeframe. In
contrast, gllvm struggles to scale beyond a few hundred variables within
practical computational limits.


# Benchmark
We compare
\begin{itemize}
\item py-PLN-CPU (PLN fitted with `pyPLNmodels` on CPU)
\item py-PLN-GPU (PLN fitted with `pyPLNmodels` on GPU)
\item py-PLN-PCA-CPU (PLN-PCA fitted with `pyPLNmodels` on CPU)
\item py-PLN-PCA-GPU (PLN-PCA fitted with `pyPLNmodels` on GPU)
\item R-PLN (PLN fitted with `PLNmodels`, on CPU)
\item R-PLN-PCA (PLN-PCA fitted with `PLNmodels`, on CPU)
\item GLLVM (on CPU)
\end{itemize}
on the `scMARK` dataset, a benchmark for scRNA data, with
$n=19998$ cells (samples) and 14059 genes (variables) are available.
We plot below the running times required to fit such models when the number of variables (i.e.
genes) grows when $n = 100,1000, 19998$. We used $q =5$ Principal Components when fitting each
PLN-PCA model and the number of latent variables LV=$2$ for the `GLLVM` model.
For each model, the fitting process was halted if the running time exceeded
10,000 seconds. We were unable to run `GLLVM` for $n = 19998$ due to CPU memory
limitations (64 GB RAM). Similarly, `py-PLN-PCA-GPU` could not be run when
$n=19998$ and $p\geq13000$ as it exceeded the GPU memory capacity (24 GB RAM).


![Running time analysis on the scMARK benchmark.](figures/plots_benchmark.pdf)
Each package uses variational inference, maximizing an Evidence
Lower BOund(ELBO) approximating the log-likelihood of the model.
```GLLVM``` uses an alternate-optimization scheme, fitting alternatively a Negative Binomial (NB) Generalized Linear
Model(GLM), and two penalized NB GLM coupled with a fixed-point algorithm, while ```pyPLNmodels``` and
```PLNmodels``` uses vanilla gradient ascent to maximize the ELBO.
```PLNmodels``` and ```GLLVM``` uses C++ backend while ```pyPLNmodels``` uses
automatic differentiation from Pytorch to compute the gradients of the ELBO. Each
PLN-PCA model is estimated using comparable variational inference methods.
However, the variational approximation for the PLN model in the
```pyPLNmodels``` version is more efficient than its counterpart in
```PLNmodels```.

<!-- # Mathematical description -->

<!-- ## Models -->

<!--  We introduce formally  the PLN [@PLN] and PLN-PCA [@PLNPCA] models. Let $n,p,d,q \in \mathbb N_{\star}^4$. We consider: -->
<!-- \begin{itemize} -->
<!-- \item $n$ samples $(i=1,\ldots,n)$ -->
<!-- \item $p$ features $(j=1,\ldots,p)$ -->
<!-- \item $n$ measures $X_{i}=\left(x_{i h}\right)_{1 \leq h \leq d}$ : -->
<!-- $X_{i h}=$ given covariate for sample $i$ -->
<!-- \item $n$  counts $Y_i = (Y_{i j})_{1\leq j \leq p}$ -->
<!-- \item $n$ offsets $O_i = (o_{ij})_{1\leq j\leq p}$ -->

<!-- \end{itemize} -->
<!-- We assume that for all ${1 \leq i \leq n}$, the observed abundances $\left(Y_{i -->
<!-- j}\right)_{1 \leq j \leq p}$ are independent conditionally on a latent variable -->
<!-- $Z_{i} \in \mathbb R^{p}$ such that: -->
<!-- \begin{equation}\label{model} -->
<!-- \begin{array}{c} -->
<!-- Z_{i} \sim \mathcal N \left(\beta^{\top}X_i, CC^{\top} \right)  \\ -->
<!--  \left(Y_{i j}  \mid Z_{i j} \right)  \sim \mathcal{P}\left(\exp \left(o_{i j} +Z_{i j}\right)\right), \\ -->
<!-- \end{array} -->
<!-- \end{equation} -->
<!--  where $\beta \in \mathbb{R}^{d \times p}$ represents the unknown regression -->
<!--  coefficients, and $C \in \mathbb{R}^{p \times q}$ denotes an unknown matrix, -->
<!--  with $q \leq p$ is a hyperparameter. When $q < p$, the model -->
<!--  corresponds to PLN-PCA. Conversely, when $q = p$, the model reverts to the -->
<!--  standard PLN. The unknown (and -->
<!--  identifiable) parameter is $\theta = (\Sigma,\beta)$, where $\Sigma = CC^{\top}$ corresponds to the covariance matrix of the gaussian component. -->

<!-- # Inference -->

<!-- We infer the parameter $\theta$ by maximizing in $(\theta, q)$ the following bi-concave Evidence Lower BOund(ELBO): -->
<!-- $$J_Y(\theta, q) = \mathbb{E}_{q}\left[\log p_{\theta}(Y, Z)\right] -\mathbb{E}_{q}[\log q(Z)] \leq \log p_{\theta}(Y),$$ -->
<!-- where $p_{\theta}$ is the model likelihood and $q=\left(q_i\right)_{1\leq i\leq -->
<!-- n}$ is a variational parameter approximating the (unknown) law $Z\mid Y$. -->


## Ongoing work
A zero-inflated version of the PLN model is currently under development, with a preprint [@PLNzero] expected to be published in the near future.

# Acknowledgements
The authors would like to thank Jean-Benoist Léger for the time spent on giving
precious advices to build a proper python package. This work was
supported by the French ANR SingleStatOmics.
<!-- # Mathematics -->

<!-- Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$ -->

<!-- Double dollars make self-standing equations: -->

<!-- $$\Theta(x) = \left\{\begin{array}{l} -->
<!-- 0\textrm{ if } x < 0\cr -->
<!-- 1\textrm{ else} -->
<!-- \end{array}\right.$$ -->

<!-- You can also use plain \LaTeX for equations -->
<!-- \begin{equation}\label{eq:fourier} -->
<!-- \hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx -->
<!-- \end{equation} -->
<!-- and refer to \autoref{eq:fourier} from text. -->

<!-- # Citations -->

<!-- Citations to entries in paper.bib should be in -->
<!-- [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html) -->
<!-- format. -->

<!-- If you want to cite a software repository URL (e.g. something on GitHub without a preferred -->
<!-- citation) then you can do it with the example BibTeX entry below for @fidgit. -->

<!-- For a quick reference, the following citation commands can be used: -->
<!-- - `@author:2001`  ->  "Author et al. (2001)" -->
<!-- - `[@author:2001]` -> "(Author et al., 2001)" -->
<!-- - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures -->

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->
<!-- and referenced from text using \autoref{fig:example}. -->

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

<!-- # Mathematical details -->
# References
