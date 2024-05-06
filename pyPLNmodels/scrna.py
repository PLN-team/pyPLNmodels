import pandas as pd
import numpy as np
import pkg_resources
import warnings

from pyPLNmodels._utils import threshold_samples_and_dim


def load_scrna(
    n_samples: int = 469,
    dim: int = 200,
    *,
    return_labels: bool = False,
    for_formula=True,
) -> np.ndarray:
    """
    Get real count data from the scMARK
    (https://zenodo.org/records/5765804) dataset.


    References:
     scMARK an ‘MNIST’ like benchmark to evaluate and optimize models for unifying scRNA data
     Swechha, Dylan Mendonca, Octavian Focsa, J. Javier Díaz-Mejía, Samuel Cooper



    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default max_samples.
    dim : int, optional
        Dimension, by default max_dim.
    return_labels: bool, optional(keyword-only)
        If True, will return the labels of the count data
    for_formula: bool, optional(keyword-only)
        If True, will return a dict so that it can
        be passed into a formula. Default is True.

    Returns
    -------
    np.ndarray
        Real count data and labels if return_labels is True.
    """
    max_samples = 469
    max_dim = 200
    n_samples, dim = threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    endog_stream = pkg_resources.resource_stream(__name__, "data/scRT/counts.csv")
    endog = pd.read_csv(endog_stream).values[:n_samples, :dim]
    best_cols = (endog > 0).mean(axis=0) > 0
    endog = endog[:, best_cols]
    print(f"Returning dataset of size {endog.shape}")
    if return_labels is False:
        if for_formula is True:
            return {"endog": endog}
        return endog
    labels_stream = pkg_resources.resource_stream(__name__, "data/scRT/labels.csv")
    labels = np.array(pd.read_csv(labels_stream).values[:n_samples].squeeze())
    if for_formula is True:
        return {"endog": endog, "labels": labels}
    return endog, labels
