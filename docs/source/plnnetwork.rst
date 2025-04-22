PlnNetwork
==========

The `PlnNetwork` model aims at inferring a network between variables.
A visualization is possible thanks to the `.viz_network()` method.
The inference is made by imposing a sparsity penalty on the
correlations (precision matrix).


See J. Chiquet, S. Robin, M. Mariadassou: *Variational Inference for
sparse network reconstruction from count data*  for more
information `[pdf] <https://proceedings.mlr.press/v97/chiquet19a.html>`_.

For an in-depth tutorial to the `PlnNetwork` model, see the
`network analysis tutorial <./vignettes/network_analysis.html>`_.

PlnNetwork Documentation
------------------------

.. autoclass:: pyPLNmodels.PlnNetwork
   :members:
   :inherited-members:
   :special-members: __init__


List of methods and attributes
------------------------------
.. autoclasstoc:: pyPLNmodels.PlnNetwork
