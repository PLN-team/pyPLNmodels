.. pyPLNmodels documentation master file, created by
   sphinx-quickstart on Tue Feb  7 19:07:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Welcome to pyPLNmodels's documentation!

API documentation
=================

PLN models encompass a range of models designed to analyze count data, all
based on the Poisson-Lognormal distribution. Each model has unique
characteristics and serves different purposes. Covariates and offsets may be included in the models.


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - :doc:`Pln <pln>` (Basic Poisson-Lognormal)
     - :doc:`PlnAR <plnar>` (AutoRegressive, or Time series/ 1D spatial)
   * - :doc:`ZIPln <zipln>` (Zero-Inflation)
     - :doc:`PlnLDA <plnlda>` (Supervised clustering)
   * - :doc:`PlnPCA <plnpca>` (Dimension reduction)
     - :doc:`PlnPCACollection <plnpcacollection>` (Dimension reduction)
   * - :doc:`PlnMixture <plnmixture>` (Unsupervised clustering)
     - :doc:`PlnMixtureCollection <plnmixturecollection>` (Unsupervised clustering)
   * - :doc:`PlnNetwork <plnnetwork>` (Network inference)
     - :doc:`PlnNetworkCollection <plnnetworkcollection>` (Network inference)
   * - :doc:`ZIPlnPCA <ziplnpca>` (Zero-Inflation and dimension reduction)
     - :doc:`ZIPlnPCACollection <ziplnpcacollection>` (Zero-Inflation and dimension reduction)
   * - :doc:`PlnDiag <plndiag>` (Diagonal covariance matrix PLN)
     -

Getting started and tutorials
=============================

A notebook to get started with pyPLNmodels is `available here
<https://github.com/PLN-team/pyPLNmodels/blob/main/Getting_started.ipynb>`_.
In-depth tutorials are `available here <https://pln-team.github.io/pyPLNmodels/tutorials/>`_.
Otherwise, the next section is a quick overview of the package.



.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/index.html


Overview
========

.. include:: ./readme.rst

.. toctree::
   :hidden:
   :caption: Models

   pln
   zipln
   plnpca
   plnpcacollection
   plnar
   plnlda
   plnnetwork
   plnnetworkcollection
   plnmixture
   plnmixturecollection
   ziplnpca
   ziplnpcacollection
   plndiag



.. toctree::
   :hidden:
   :caption: Overview

   readme


.. toctree::
   :hidden:
   :caption: Links

   ↪ PyPI <https://pypi.org/project/pyPLNmodels/>
   ↪ GitHub <https://github.com/PLN-team/pyPLNmodels>
   ↪ Tutorials <https://pln-team.github.io/pyPLNmodels/tutorials/>
