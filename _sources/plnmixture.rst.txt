PlnMixture
==========

The `PlnMixture` model clusters count data. Clusters can be accessed through
the `.clusters` attribute after fitting the model (`.fit()` method). The use of covariates is
possible, but the regression coefficient is shared among all clusters. The performance may decrease significantly with the number of covariates.
Note that the number of clusters is a hyperparameter that needs to be set by the
user.

For an in-depth tutorial to the `PlnMixture` model, see the
`clustering tutorial <./vignettes/clustering.html>`_.

PlnMixture Documentation
------------------------

.. autoclass:: pyPLNmodels.PlnMixture
   :members:
   :inherited-members:
   :special-members: __init__


List of methods and attributes
------------------------------
.. autoclasstoc:: pyPLNmodels.PlnMixture
