import pkg_resources
import pandas as pd
import numpy as np

from .utils import _threshold_samples_and_dim


def load_scrna(
    n_samples: int = 400,
    dim: int = 100,
) -> np.ndarray:
    """
    Get real count data from the scMARK (https://zenodo.org/records/5765804) dataset.
    Only a tiny subset of the dataset is available, for memory purposes.


    References:
     scMARK an ‘MNIST’ like benchmark to evaluate and optimize models for unifying scRNA data
     Swechha, Dylan Mendonca, Octavian Focsa, J. Javier Díaz-Mejía, Samuel Cooper



    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 100. Can not be greater than 400.
    dim : int, optional
        Dimension (i.e. number of variables), by default 100.
        Can not be greater than 100.

    Returns
    -------
    Dict
        Dictionary with three keys: "endog", "labels" and "labels_value".
        "endog" is a matrix of counts, "labels" is an array of cell types that
        are one-hot encoded and "labels_value" gives the actual cell type,
        either "T_cells_CD4+" or "T_cells_CD8+".
    """
    max_samples = 400
    max_dim = 100
    n_samples, dim = _threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    endog = pd.read_csv(pkg_resources.resource_stream(__name__, "data/scRT/counts.csv"))
    print(f"Returning scRNA dataset of size {endog.shape}")
    cell_type = pd.read_csv(
        pkg_resources.resource_stream(__name__, "data/scRT/labels.csv")
    )
    labels = pd.get_dummies(cell_type)

    return {"endog": endog, "labels": labels, "labels_value": cell_type}
