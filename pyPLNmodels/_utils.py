from functools import wraps
import math
import textwrap
import time

import matplotlib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import seaborn as sns
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_log_sum_of_endog(endog: torch.Tensor) -> torch.Tensor:
    """
    Compute offsets from the sum of `endog`.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)

    Returns
    -------
    torch.Tensor
        Offsets of size (n, p)
    """
    sum_of_endog = torch.sum(endog, axis=1)
    return torch.log(sum_of_endog.repeat((endog.shape[1], 1))).T


class _TimeRecorder:  # pylint: disable=too-few-public-methods
    def __init__(self, time_to_remove_from_beginning):
        self.running_times = []
        self._beginning_time = time.time() - time_to_remove_from_beginning

    def track_running_time(self):
        """Track the running time since the fitting of the model has been launched."""
        self.running_times.append(time.time() - self._beginning_time)


def _log_stirling(integer: torch.Tensor) -> torch.Tensor:
    """
    Compute log(n!) using the Stirling formula.

    Parameters
    ----------
    integer : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Approximation of log(n!) element-wise.
    """
    integer_ = integer + (integer == 0)  # Replace 0 with 1 since 0! = 1!
    return torch.log(torch.sqrt(2 * np.pi * integer_)) + integer_ * torch.log(
        integer_ / math.exp(1)
    )


def _add_doc(
    parent_class,
    *,
    params=None,
    example=None,
    returns=None,
    see_also=None,
    raises=None,
    notes=None,
):  # pylint: disable=too-many-arguments
    def wrapper(fun):
        # if isinstance(fun, classmethod):
        #     fun = fun.__func__

        doc = getattr(parent_class, fun.__name__).__doc__
        doc = textwrap.dedent(doc).rstrip(" \n\r")
        if params is not None:
            doc += textwrap.dedent(params.rstrip(" \n\r"))
        if returns is not None:
            doc += "\n\nReturns"
            doc += "\n-------"
            doc += textwrap.dedent(returns)
        if see_also is not None:
            doc += "\n\nSee also"
            doc += "\n--------"
            doc += textwrap.dedent(see_also)
        if example is not None:
            doc += "\n\nExamples"
            doc += "\n--------"
            doc += textwrap.dedent(example)
        if raises is not None:
            doc += "\n\nRaises"
            doc += "\n------"
            doc += textwrap.dedent(raises)
        if notes is not None:
            doc += "\n\nNotes"
            doc += "\n-----"
            doc += textwrap.dedent(notes)
        fun.__doc__ = doc
        return fun

    return wrapper


def _nice_string_of_dict(dictionnary: dict, best_rank: int = None) -> str:
    """
    Create a nicely formatted string representation of a dictionary,
    optionally highlighting the best model.

    Parameters
    ----------
    dictionnary : dict
        Dictionary to format.
    best_rank : int, optional
        The rank of the best model to highlight, by default None.

    Returns
    -------
    str
        Nicely formatted string representation of the dictionary.
    """
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            if best_rank is not None and element == best_rank:
                return_string += f"{str(element):>12}**"
            else:
                return_string += f"{str(element):>12}"
        return_string += "\n"
    return return_string


def calculate_correlation(data, transformed_data):
    """
    Calculate correlations between each variable in `data` and the first two principal components.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix with shape (n_samples, n_features).
    transformed_data : np.ndarray
        Data matrix after PCA transformation.

    Returns
    -------
    ccircle : list of tuples
        List of tuples containing correlations with the first and second principal components.
    """
    ccircle = []
    for j in data.T:
        corr1 = np.corrcoef(j, transformed_data[:, 0])[0, 1]
        corr2 = np.corrcoef(j, transformed_data[:, 1])[0, 1]
        ccircle.append((corr1, corr2))
    return ccircle


def _trunc_log(tens: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    integer = torch.min(
        torch.max(tens, torch.tensor([eps], device=DEVICE)),
        torch.tensor([1 - eps], device=DEVICE),
    )
    return torch.log(integer)


def _log1pexp(t):
    mask = t > 10
    return torch.where(
        mask,
        t,
        torch.log(1 + torch.exp(t)),
    )


def _process_indices_of_variables(
    variables_names, indices_of_variables, column_names_endog
):
    if indices_of_variables is None:
        indices_of_variables = [
            column_names_endog.get_loc(name) for name in variables_names
        ]
    else:
        if len(indices_of_variables) != len(variables_names):
            raise ValueError(
                f"Number of indices ({len(indices_of_variables)}) should be "
                f"the same as the number of variable names ({len(variables_names)})."
            )
    return indices_of_variables


def _shouldbefitted(func):
    """
    Decorator to check if the model has been fitted before executing the function.
    Raises a `RuntimeError` if the model is not fitted.
    """

    @wraps(func)
    def _func(self, *args, **kwargs):
        if self._fitted is False:  # pylint: disable=protected-access
            raise RuntimeError("Please fit the model before.")
        return func(self, *args, **kwargs)

    return _func


def _none_if_no_exog(func):
    @wraps(func)
    def _func(self, *args, **kwargs):
        if self.nb_cov == 0:
            print("No exog in the model, so no coefficients. Returning None.")
            return None
        return func(self, *args, **kwargs)

    return _func


def _two_dim_latent_variances(components, latent_sqrt_variance):
    components_var = np.expand_dims(latent_sqrt_variance**2, 1) * np.expand_dims(
        components, 0
    )
    covariances = np.matmul(components_var, np.expand_dims(components.T, 0))
    return covariances


def get_label_mapping(cluster_labels: ArrayLike, true_labels: ArrayLike):
    """
    Generate a mapping from cluster labels to true labels based on majority voting.

    Parameters
    ----------
    cluster_labels : array-like of shape (n_samples,)
        Cluster labels assigned by the clustering algorithm.
    true_labels : array-like of shape (n_samples,)
        True labels of the samples.

    Returns
    -------
    label_mapping : dict
        Dictionary where keys are cluster labels and values are the most frequent true label
        within each cluster. If a cluster has no true labels, it is mapped to -1.
    """
    label_mapping = {}
    n_clusters = len(np.unique(cluster_labels))
    for cluster in range(n_clusters):
        mask = cluster_labels == cluster
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        if np.sum(mask) > 0:  # Check if there are any true labels for this cluster
            majority_class = mode(true_labels[mask], keepdims=True)[0][0]
            label_mapping[cluster] = majority_class
        else:
            label_mapping[cluster] = -1
    return label_mapping


def get_confusion_matrix(pred_clusters: ArrayLike, true_clusters: ArrayLike):
    """
    Compute the confusion matrix for k-means clustering results.

    Parameters
    ----------
    pred_clusters : array-like of shape (n_samples,)
        Predicted cluster labels from k-means clustering.
    true_clusters : array-like of shape (n_samples,)
        True cluster labels.

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] is the number of samples with true label being i-th class
        and predicted label being j-th class.
    """
    label_mapping = get_label_mapping(pred_clusters, true_clusters)
    mapped_labels = np.vectorize(lambda x: label_mapping.get(x, -1))(pred_clusters)
    cm = confusion_matrix(true_clusters, mapped_labels)
    return cm


def plot_confusion_matrix(
    confusion_mat: np.ndarray, ax: matplotlib.axes.Axes, title: str = ""
):
    """
    Plot the confusion matrix using a heatmap.

    Parameters
    ----------
    confusion_mat : ndarray of shape (n_classes, n_classes)
        Confusion matrix to be plotted.
    ax : matplotlib.axes.Axes
        Axes object to draw the heatmap on.
    title : str (Optional)
        Title for the heatmap.
    """
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)
