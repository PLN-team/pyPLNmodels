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
        Number of samples, by default 400. Cannot be greater than 2000.
    dim : int, optional
        Dimension (i.e. number of variables, genes), by default 100.
        Cannot be greater than 3000.

    Returns
    -------
    Dict
        Dictionary with three keys: `endog`, `labels`, and `labels_1hot`.
            `endog` is a matrix of counts.
            `labels` is an array giving cell types with three possibilities,
                    either `T_cells_CD4+` or `T_cells_CD8+` or `Macrophages`.
            `labels_1hot` corresponds to a 2D array that is the
                one-hot encoding of the `labels` array.

    Examples
    --------
    >>> from pyPLNmodels import load_scrna
    >>> scrna = load_scrna()
    >>> print('Keys: ', scrna.keys())
    >>> print(scrna["endog"].head())
    >>> print(scrna["endog"].describe())
    """
    max_samples = 2000
    max_dim = 3000
    n_samples, dim = _threshold_samples_and_dim(max_samples, max_dim, n_samples, dim)
    endog = pd.read_csv(
        pkg_resources.resource_stream(__name__, "data/scRT/counts.csv")
    ).iloc[:n_samples, :dim]
    print(f"Returning scRNA dataset of size {endog.shape}")
    cell_type = pd.read_csv(
        pkg_resources.resource_stream(__name__, "data/scRT/labels.csv")
    ).squeeze()[:n_samples]
    labels_1hot = pd.get_dummies(cell_type)
    return {"endog": endog, "labels": cell_type, "labels_1hot": labels_1hot}
