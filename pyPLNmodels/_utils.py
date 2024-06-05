import os
import math
import warnings
import textwrap
from typing import Optional, Dict, Any, Union, Tuple, List

import numpy as np
from patsy import PatsyError
import pandas as pd
import torch
import torch.linalg as TLA
from matplotlib import transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices, dmatrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Circle

from pyPLNmodels.lambert import lambertw


torch.set_default_dtype(torch.float64)


if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

BETA = 0.03


class _CriterionArgs:
    def __init__(self):
        """
        Initialize the PlotArgs class.

        Parameters
        ----------
        window : int
            The size of the window for computing the criterion.
        """
        self.running_times = []
        self._elbos_list = []
        self.cumulative_elbo_list = [0]
        self.new_derivative = 0
        self.normalized_elbo_list = []
        self.criterion_list = [1]
        self.criterion = 1
        # self.previous_elbo = 1

    def update_criterion(self, elbo, running_time):
        self._elbos_list.append(elbo)
        self.running_times.append(running_time)
        self.cumulative_elbo_list.append(self.cumulative_elbo + elbo)
        self.normalized_elbo_list.append(-elbo / self.cumulative_elbo_list[-1])
        if self.iteration_number > 1:
            current_derivative = np.abs(
                (self.normalized_elbo_list[-2] - self.normalized_elbo_list[-1])
            )
            old_derivative = self.new_derivative
            self.new_derivative = (
                self.new_derivative * (1 - BETA) + current_derivative * BETA
            )
            current_hessian = np.abs(
                (self.new_derivative - old_derivative)
                / (self.running_times[-2] - self.running_times[-1])
            )
            self.criterion = self.criterion * (1 - BETA) + current_hessian * BETA
            # self.criterion = np.abs((elbo - self.previous_elbo)/elbo)
            # self.previous_elbo = elbo

            self.criterion_list.append(self.criterion)

    @property
    def iteration_number(self) -> int:
        """
        Numer of iterations done when fitting the model.

        Returns
        -------
        int
            The number of iterations.
        """
        return len(self._elbos_list)

    @property
    def cumulative_elbo(self):
        return self.cumulative_elbo_list[-1]

    def _show_loss(self, ax=None):
        """
        Show the loss of the model (i.e. the negative ELBO).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, will be created.
        """
        ax = plt.gca() if ax is None else ax
        ax.plot(self.running_times, -np.array(self._elbos_list), label="Negative ELBO")
        last_elbos = np.round(self._elbos_list[-1], 6)
        ax.set_title(f"Negative ELBO. Best ELBO ={last_elbos}")
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("ELBO")
        ax.legend()

    def _show_stopping_criterion(self, ax=None):
        """
        Show the stopping criterion plot. The gradient ascent
        stops according to this critertion.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, will be created.
        """
        ax = plt.gca() if ax is None else ax
        ax.plot(
            self.running_times,
            self.criterion_list,
            label="Delta",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Delta")
        ax.set_title("Increments")
        ax.legend()


def _sigmoid(tens: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-tens))


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
    return (
        integer_ * torch.log(integer_)
        - integer_
        + torch.log(8 * integer_**3 + 4 * integer_**2 + integer_ + 1 / 30) / 6
        + math.log(math.pi) / 2
    )

    return torch.log(torch.sqrt(2 * np.pi * integer_)) + integer_ * torch.log(
        integer_ / math.exp(1)
    )


def _trunc_log(tens: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    integer = torch.min(
        torch.max(tens, torch.tensor([eps], device=DEVICE)),
        torch.tensor([1 - eps], device=DEVICE),
    )
    return torch.log(integer)


def _get_offsets_from_sum_of_endog(endog: torch.Tensor) -> torch.Tensor:
    """
    Compute offsets from the sum of endog.

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
    return sum_of_endog.repeat((endog.shape[1], 1)).T


def _raise_wrong_dimension_error(
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_order_first: int,
    dim_order_second: int,
) -> None:
    """
    Raise an error for mismatched dimensions between two tensors.

    Parameters
    ----------
    str_first_array : str
        Name of the first tensor
    str_second_array : str
        Name of the second tensor
    dim_first_array : int
        Dimension of the first tensor
    dim_second_array : int
        Dimension of the second tensor
    dim_order_first : int
        Dimension causing the error for the first tensor.
    dim_order_second : int
        Dimension causing the error for the second tensor.
    Raises
    ------
    ValueError
        If the dimensions of the two tensors do not match.
    """
    msg = (
        f"The size of tensor {str_first_array} at non-singleton dimension {dim_order_first} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_order_second}"
    )
    raise ValueError(msg)


def _format_data(
    data: Union[torch.Tensor, np.ndarray, pd.DataFrame]
) -> torch.Tensor or None:
    """
    Transforms the data in a torch.tensor if the input is an array, and None if the input is None.
    Raises an error if the input is not an array or None.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or torch.Tensor
        Input data.

    Returns
    -------
    torch.Tensor or None
        Formatted data.

    Raises
    ------
    AttributeError
        If the value is not an array or None.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).double()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).double()
    if isinstance(data, torch.Tensor):
        return data
    raise AttributeError(
        "Please insert either a numpy.ndarray, pandas.DataFrame or torch.Tensor"
    )


def _format_model_param(
    endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    exog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    offsets_formula: str,
    take_log_offsets: bool,
    add_const: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Format each of the model parameters to an array or None if None.

    Parameters
    ----------
    endog : Union[torch.Tensor, np.ndarray, pd.DataFrame], shape (n, )
        Count data.
    exog : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None, shape (n, d) or None
        Covariate data.
    offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None, shape (n, ) or None
        Offset data.
    offsets_formula : str
        Formula for calculating offsets.
    take_log_offsets : bool
        Flag indicating whether to take the logarithm of offsets.
    add_const: bool
        Whether to add a column of one in the exog.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Formatted model parameters.

    Raises
    ------
    ValueError
        If endog has negative values or offsets_formula is not None and not "logsum" or "zero"
        If endog has one line that is full of zeros.
    """
    endog = _format_data(endog)
    if torch.min(endog) < 0:
        raise ValueError("Counts should be only non negative values.")
    exog = _format_data(exog)
    if add_const is True:
        exog = _add_const_to_exog(exog, axis=0, length=endog.shape[0])
    if exog is not None:
        _check_full_rank_exog(exog)
    if offsets is None:
        if offsets_formula == "logsum":
            print("Setting the offsets as the log of the sum of endog")
            offsets = torch.log(_get_offsets_from_sum_of_endog(endog)).double()
        elif offsets_formula == "zero":
            print("Setting the offsets to zero")
            offsets = torch.zeros(endog.shape)
        else:
            raise ValueError(
                'Wrong offsets_formula. Expected either "zero" or "logsum", got {offsets_formula}'
            )
    else:
        offsets = _format_data(offsets)
        if take_log_offsets is True:
            offsets = torch.log(offsets)
    return endog, exog, offsets


def _has_null_variance(tensor: torch.Tensor) -> bool:
    """
    Check if a torch.Tensor has a dimension with null variance.

    Parameters
    ----------
        tensor (torch.Tensor): The input tensor.

    Returns
    -------
        bool: True if a dimension with null variance is found, False otherwise.
    """
    variances = torch.var(tensor, dim=0)
    has_null_var = torch.any(variances == 0)
    return bool(has_null_var)


def _check_data_shape(
    endog: torch.Tensor, exog: torch.Tensor, offsets: torch.Tensor
) -> None:
    """
    Check if the shape of the input data is valid.

    Parameters
    ----------
    endog : torch.Tensor, shape (n, p)
        Count data.
    exog : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, p) or None
        Offset data.
    """
    n_endog, p_endog = endog.shape
    n_offsets, p_offsets = offsets.shape
    _check_two_tensors_dimensions_are_equal(
        "endog", "offsets", n_endog, n_offsets, 0, 0
    )
    if exog is not None:
        n_cov, _ = exog.shape
        _check_two_tensors_dimensions_are_equal("endog", "exog", n_endog, n_cov, 0, 0)
    _check_two_tensors_dimensions_are_equal(
        "endog", "offsets", p_endog, p_offsets, 1, 1
    )


def _nice_string_of_dict(dictionnary: dict) -> str:
    """
    Create a nicely formatted string representation of a dictionary.

    Parameters
    ----------
    dictionnary : dict
        Dictionary to format.

    Returns
    -------
    str
        Nicely formatted string representation of the dictionary.
    """
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            return_string += f"{str(element):>12}"
        return_string += "\n"
    return return_string


def _plot_ellipse(mean_x: float, mean_y: float, cov: np.ndarray, ax) -> float:
    """
    Plot an ellipse given two coordinates and the covariance.

    Parameters:
    -----------
    mean_x : float
        x-coordinate of the mean.
    mean_y : float
        y-coordinate of the mean.
    cov : np.ndarray
        Covariance matrix of the 2d vector.
    ax : object
        Axes object to plot the ellipse on.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        linestyle="--",
        alpha=0.2,
    )

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def _check_2d_tens_right_shape(array, expected, array_name):
    shape = array.shape
    if shape[0] != expected[0]:
        msg = f" should have dimension {expected[0]} at dimension 0, got {shape[0]}."
        raise ValueError(msg)
    if shape[1] != expected[1]:
        msg = f"{array_name} should have dimension {expected[1]} at dimension 1, got {shape[1]}."
        raise ValueError(msg)


def _check_two_tensors_dimensions_are_equal(
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_order_first: int,
    dim_order_second: int,
) -> None:
    """
    Check if two dimensions are equal.

    Parameters
    ----------
    str_first_array : str
        Name of the first array.
    str_second_array : str
        Name of the second array.
    dim_first_array : int
        Dimension of the first array.
    dim_second_array : int
        Dimension of the second array.
    dim_order_first : int
        Dimension causing the error for the first tensor.
    dim_order_second : int
        Dimension causing the error for the second tensor.

    Raises
    ------
    ValueError
        If the dimensions of the two arrays are not equal.
    """
    if dim_first_array != dim_second_array:
        _raise_wrong_dimension_error(
            str_first_array,
            str_second_array,
            dim_first_array,
            dim_second_array,
            dim_order_first,
            dim_order_second,
        )


def _check_right_rank(data: Dict[str, Any], rank: int) -> None:
    """
    Check if the rank of the given data matches the specified rank.

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing the data.
    rank : int
        The expected rank.

    Raises
    ------
    RuntimeError
        If the rank of the data does not match the specified rank.

    """
    data_rank = data["latent_mean"].shape[1]
    if data_rank != rank:
        raise RuntimeError(
            f"Wrong rank during initialization. Got rank {rank} and data with rank {data_rank}."
        )


def _extract_data_from_formula_with_infla(
    formula: str, data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]]
) -> Tuple:
    """
    Extract data from the given formula and data dictionary. Also Deal with
    inflation also.

    Parameters
    ----------
    formula : str
        The formula specifying the data to extract.
    data : Dict[str, Any]
        A dictionary containing the data.

    Returns
    -------
    Tuple
        A tuple containing the extracted endog, exog, exog_infla and offsets.
    """
    split_formula = formula.split("|")
    formula_exog = split_formula[0]
    endog, exog, offsets = _extract_data_from_formula_no_infla(formula_exog, data)
    formula_infla = split_formula[1]
    try:
        exog_infla = dmatrix(formula_infla, data=data)
    except PatsyError as err:
        msg = "Formula of exog infla did not work: {formula_infla}."
        msg += " Falling back to an intercept. Error from Patsy:"
        warnings.warn(msg)
        print(err)
        return endog, exog, None, offsets
    non_zero = ((exog_infla) ** 2).sum(axis=0) > 0
    exog_infla = exog_infla[:, non_zero]
    return endog, exog, exog_infla, offsets


def _extract_data_from_formula_no_infla(
    formula: str, data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]]
) -> Tuple:
    """
    Extract data from the given formula and data dictionary.

    Parameters
    ----------
    formula : str
        The formula specifying the data to extract.
    data : Dict[str, Any]
        A dictionary containing the data.

    Returns
    -------
    Tuple
        A tuple containing the extracted endog, exog, and offsets.

    """
    variables = dmatrices(formula, data=data)
    endog = variables[0]
    exog = variables[1]
    non_zero = exog.sum(axis=0) > 0
    exog = exog[:, non_zero]

    if exog.size == 0:
        exog = None
    if "offsets" in data.keys():
        offsets = data["offsets"]
        print("Taking the offsets from the data given.")
    else:
        offsets = None
    return endog, exog, offsets


def _is_dict_of_dict(dictionary: Dict[Any, Any]) -> bool:
    """
    Check if the given dictionary is a dictionary of dictionaries.

    Parameters
    ----------
    dictionary : Dict[Any, Any]
        The dictionary to check.

    Returns
    -------
    bool
        True if the dictionary is a dictionary of dictionaries, False otherwise.

    """
    return isinstance(dictionary[List(dictionary.keys())[0]], dict)


def _get_dict_initialization(
    rank: int, dict_of_dict: Optional[Dict[int, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """
    Get the initialization dictionary for the given rank.

    Parameters
    ----------
    rank : int
        The rank to get the initialization dictionary for.
    dict_of_dict : Dict[int, Dict[str, Any]], optional
        A dictionary containing initialization dictionaries for different ranks.

    Returns
    -------
    Optional[Dict[str, Any]]
        The initialization dictionary for the given rank, or None if it does not exist.

    """
    if dict_of_dict is None:
        return None
    return dict_of_dict.get(rank)


def _to_tensor(
    obj: Union[np.ndarray, torch.Tensor, pd.DataFrame]
) -> Union[torch.Tensor]:
    """
    Convert an object to a PyTorch tensor.

    Parameters
    ----------
        obj (np.ndarray or torch.Tensor or pd.DataFrame or None):
            The object to be converted.

    Returns
    -------
        torch.Tensor or None:
            The converted PyTorch tensor.

    Raises
    ------
        TypeError:
            If the input object is not an np.ndarray, torch.Tensor, pd.DataFrame, or None.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, pd.DataFrame):
        return torch.from_numpy(obj.values)
    raise TypeError(
        "Please give either an np.ndarray or torch.Tensor or pd.DataFrame or None"
    )


def _array2tensor(func):
    def setter(self, array_like):
        array_like = _to_tensor(array_like)
        func(self, array_like)

    return setter


def _handle_data_with_inflation(
    endog,
    exog,
    exog_inflation,
    offsets,
    offsets_formula,
    zero_inflation_formula,
    take_log_offsets,
    add_const,
    add_const_inflation,
    batch_size,
):
    (
        endog,
        exog,
        offsets,
        column_endog,
        samples_only_zeros,
        dim_only_zeros,
        batch_size,
    ) = _handle_data(
        endog, exog, offsets, offsets_formula, take_log_offsets, add_const, batch_size
    )
    ## changing dimension if row-wise and a vector of ones
    exog_inflation = _format_data(exog_inflation)
    exog_inflation, add_const_inflation = _get_coherent_inflation_inits(
        zero_inflation_formula, exog_inflation, add_const_inflation
    )
    if exog_inflation is not None:
        if zero_inflation_formula == "column-wise":
            _check_full_rank_exog(exog_inflation, inflation=True)
        elif zero_inflation_formula == "row-wise":
            _check_full_rank_exog(exog_inflation.T, inflation=True)

    if zero_inflation_formula == "row-wise" and exog_inflation is not None:
        if torch.count_nonzero(exog_inflation - 1) == 0:
            exog_inflation = torch.ones(1, endog.shape[1])
    if exog_inflation is not None:
        if zero_inflation_formula == "column-wise":
            exog_inflation = exog_inflation[~samples_only_zeros, :]
        elif zero_inflation_formula == "row-wise":
            exog_inflation = exog_inflation[:, ~dim_only_zeros]

    if zero_inflation_formula != "global" and exog_inflation is not None:
        _check_shape_exog_infla(
            exog_inflation, zero_inflation_formula, endog.shape[0], endog.shape[1]
        )
    if zero_inflation_formula == "global":
        exog_inflation = None
    else:
        if add_const_inflation is True:
            if zero_inflation_formula == "column-wise":
                exog_inflation = _add_const_to_exog(exog_inflation, 0, endog.shape[0])
            else:
                exog_inflation = _add_const_to_exog(exog_inflation, 1, endog.shape[1])

    dirac = endog == 0
    onlyonebatch = batch_size == (endog.shape[0])
    if onlyonebatch is True:
        if exog_inflation is not None:
            exog_inflation = exog_inflation.to(DEVICE)
        dirac = dirac.to(DEVICE)
    return (
        endog,
        exog,
        exog_inflation,
        offsets,
        column_endog,
        dirac,
        batch_size,
        samples_only_zeros,
    )


def _remove_samples(endog, exog, offsets, samples_only_zeros):
    endog = endog[~samples_only_zeros, :]
    if exog is not None:
        exog = exog[~samples_only_zeros]
    offsets = offsets[~samples_only_zeros]
    return endog, exog, offsets


def _remove_dims(endog, exog, offsets, dims_only_zeros):
    endog = endog[:, ~dims_only_zeros]
    offsets = offsets[:, ~dims_only_zeros]
    return endog, exog, offsets


def _handle_batch_size(batch_size, n_samples):
    if batch_size is None:
        batch_size = n_samples

    if batch_size > n_samples:
        raise ValueError(
            f"batch_size ({batch_size}) can not be greater than the number of samples ({n_samples})"
        )
    elif isinstance(batch_size, int) is False:
        raise ValueError(f"batch_size should be int, got {type(batch_size)}")
    return batch_size


def _handle_data(
    endog,
    exog,
    offsets,
    offsets_formula: str,
    take_log_offsets: bool,
    add_const: bool,
    batch_size: bool,
) -> tuple:
    """
    Handle the input data for the model.

    Parameters
    ----------
        endog : The endog data. If a DataFrame is provided, the column names are stored for later use.
        exog : The exog data.
        offsets : The offsets data.
        offsets_formula : The formula used for offsets.
        take_log_offsets : Indicates whether to take the logarithm of the offsets.
        add_const : Indicates whether to add a constant column to the exog.
        batch_size: int. Raises an error if greater than endog.shape[0]

    Returns
    -------
        tuple: A tuple containing the processed endog, exog, offsets, and column endog (if available).

    Raises
    ------
        ValueError: If the shapes of endog, exog, and offsets do not match, or the batch_size is
        greater than endog.shape[0]
    """
    if isinstance(endog, pd.DataFrame):
        column_endog = endog.columns
    else:
        column_endog = None
    endog, exog, offsets = _format_model_param(
        endog, exog, offsets, offsets_formula, take_log_offsets, add_const
    )
    _check_data_shape(endog, exog, offsets)
    samples_only_zeros = torch.sum(endog, axis=1) == 0
    if torch.sum(samples_only_zeros) > 0.5:
        samples = torch.arange(endog.shape[0])[samples_only_zeros]
        msg = f"The ({len(samples)}) following (index) counts contains only zeros and are removed."
        msg += str(samples.numpy())
        msg += "You can access the samples that are non zeros with .useful_indices"
        warnings.warn(msg)
        endog, exog, offsets = _remove_samples(endog, exog, offsets, samples_only_zeros)
        print(f"Now dataset of size {endog.shape}")
    dim_only_zeros = torch.sum(endog, axis=0) == 0
    if torch.sum(dim_only_zeros) > 0.5:
        dims = torch.arange(endog.shape[1])[dim_only_zeros]
        msg = f"The ({len(dims)}) following (index) variables contains only zeros and are removed."
        msg += str(dims.numpy())
        warnings.warn(msg)
        endog, exog, offsets = _remove_dims(endog, exog, offsets, dim_only_zeros)
        print(f"Now dataset of size {endog.shape}")
    n_samples = endog.shape[0]
    batch_size = _handle_batch_size(batch_size, n_samples)
    if batch_size == n_samples:
        endog = endog.to(DEVICE)
        if exog is not None:
            exog = exog.to(DEVICE)
        offsets = offsets.to(DEVICE)
    return (
        endog,
        exog,
        offsets,
        column_endog,
        samples_only_zeros,
        dim_only_zeros,
        batch_size,
    )


def _add_doc(parent_class, *, params=None, example=None, returns=None, see_also=None):
    def wrapper(fun):
        doc = getattr(parent_class, fun.__name__).__doc__
        if doc is None:
            doc = ""
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
        fun.__doc__ = doc
        return fun

    return wrapper


def point_fixe_lambert(x, y):
    return x - (1 - (y * torch.exp(-x) + 1) / (x + 1))


def lambert(y, nb_pf=10):
    x = torch.log(1 + y)
    for _ in range(nb_pf):
        x = point_fixe_lambert(x, y)
    return x


def d_varpsi_x1(mu, sigma2):
    W = lambert(sigma2 * torch.exp(mu))
    first = phi(mu, sigma2)
    third = 1 / sigma2 + 1 / 2 * 1 / ((1 + W) ** 2)
    return -first * W * third


def phi(mu, sigma2):
    y = sigma2 * torch.exp(mu)
    lamby = lambertw(y)
    # lamby = lambert(y)
    log_num = -1 / (2 * sigma2) * (lamby**2 + 2 * lamby)
    return torch.exp(log_num) / torch.sqrt(1 + lamby)


def d_varpsi_x2(mu, sigma2):
    first = d_varpsi_x1(mu, sigma2) / sigma2
    W = lambert(sigma2 * torch.exp(mu))
    second = (W**2 + 2 * W) / 2 / (sigma2**2) * phi(mu, sigma2)
    return first + second


def d_h_x2(a, x, y, dirac):
    rho = torch.sigmoid(a - torch.log(phi(x, y))) * dirac
    rho_prime = rho * (1 - rho)
    return -rho_prime * d_varpsi_x1(x, y) / phi(x, y)


def d_h_x3(a, x, y, dirac):
    rho = torch.sigmoid(a - torch.log(phi(x, y))) * dirac
    rho_prime = rho * (1 - rho)
    return -rho_prime * d_varpsi_x2(x, y) / phi(x, y)


def vec_to_mat(C, p, q):
    c = torch.zeros(p, q)
    c[torch.tril_indices(p, q, offset=0).tolist()] = C
    # c = C.reshape(p,q)
    return c


def mat_to_vec(matc, p, q):
    tril = torch.tril(matc)
    # tril = matc.reshape(-1,1).squeeze()
    return tril[torch.tril_indices(p, q, offset=0).tolist()]


def _log1pexp(t):
    mask = t > 10
    return torch.where(
        mask,
        t,
        torch.log(1 + torch.exp(t)),
    )


def calculate_correlation(X, Xpca):
    """
    Calculate correlations between each variable in X and the first two principal components.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix with shape (n_samples, n_features).
    Xpca : np.ndarray
        Data matrix after PCA transformation.

    Returns
    -------
    ccircle : list of tuples
        List of tuples containing correlations with the first and second principal components.
    """
    ccircle = []
    for j in X.T:
        corr1 = np.corrcoef(j, Xpca[:, 0])[0, 1]
        corr2 = np.corrcoef(j, Xpca[:, 1])[0, 1]
        ccircle.append((corr1, corr2))
    return ccircle


def plot_correlation_arrows(axs, ccircle, variables_names):
    """
    Plot arrows representing the correlation circle.

    Parameters
    ----------
    axs : matplotlib.axes._axes.Axes
        Axes object for plotting.
    ccircle : list of tuples
        List of tuples containing correlations with the first and second principal components.
    variables_names : list
        List of names for the variables corresponding to columns in X.

    Returns
    -------
    None
    """
    for i, (corr1, corr2) in enumerate(ccircle):
        axs.arrow(
            0,
            0,
            corr1,  # 0 for PC1
            corr2,  # 1 for PC2
            lw=2,  # line width
            length_includes_head=True,
            head_width=0.05,
            head_length=0.05,
        )
        axs.text(corr1 / 2, corr2 / 2, variables_names[i])


def plot_correlation_circle(X, variables_names, indices_of_variables, title=""):
    """
    Plot a correlation circle for principal component analysis (PCA).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix with shape (n_samples, n_features).
    variables_names : list
        List of names for the variables corresponding to columns in X.
    indices_of_variables : list
        List of indices of the variables to be considered in the plot.
    title : str
        Additional title on the plot.

    Returns
    -------
    None
    """
    Xstd = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xstd)
    explained_ratio = pca.explained_variance_ratio_

    ccircle = calculate_correlation(X[:, indices_of_variables], Xpca)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(figsize=(6, 6))
    plot_correlation_arrows(axs, ccircle, variables_names)

    # Draw the unit circle, for clarity
    circle = Circle((0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5)
    axs.add_patch(circle)
    axs.set_xlabel(f"PCA 1 {(np.round(explained_ratio[0]*100, 3))}%")
    axs.set_ylabel(f"PCA 2 {(np.round(explained_ratio[1]*100, 3))}%")
    axs.set_title(f"Correlation circle on the transformed variables{title}")
    # plt.ion()

    # plt.tight_layout()
    plt.show()


def _check_formula(zero_inflation_formula):
    list_available = ["column-wise", "row-wise", "global"]
    if zero_inflation_formula not in list_available:
        msg = f"Wrong inflation formula, got {zero_inflation_formula}, expected one of {list_available}"
        raise ValueError(msg)


def _add_const_to_exog(exog, axis, length):
    if axis == 0:
        dim_concat = 1
        ones = torch.ones(length, 1)
        has_null_var = _has_null_variance(exog) if exog is not None else None
    elif axis == 1:
        dim_concat = 0
        ones = torch.ones(1, length)
        has_null_var = _has_null_variance(exog.T) if exog is not None else None
    if has_null_var is False:
        exog = torch.concat((exog, ones), dim=dim_concat)
    elif has_null_var is None:
        exog = ones
    return exog


def _get_coherent_inflation_inits(
    inflation_formula, exog_inflation, add_const_inflation
):
    if inflation_formula in {"column-wise", "row-wise"}:
        if exog_inflation is None and add_const_inflation is False:
            msg = "No exog_inflation has been given and the "
            msg += f"zero_inflation_formula is set to {inflation_formula}. "
            msg += "add_const_inflation is set to True since a ZIPln"
            msg += "must have at least an intercept for the exog_inflation."
            warnings.warn(msg)
            add_const_inflation = True
    else:
        if exog_inflation is not None:
            msg = "exog_inflation useless as zero_inflation_formula is"
            msg += " global. exog_inflation set to None"
            warnings.warn(msg)
            exog_inflation = None
        if add_const_inflation is True:
            msg = "add_const_inflation=True useless as zero_inflation_formula is"
            msg += " global. Set to False."
            warnings.warn(msg)
            add_const_inflation = False
    return exog_inflation, add_const_inflation


def _check_shape_exog_infla(exog_inflation, inflation_formula, n_samples, dim):
    if inflation_formula == "column-wise":
        if exog_inflation.shape[0] != n_samples:
            msg = "Your formula inflation is {inflation_formula}."
            msg = f"exog_inflation should have shape [{n_samples},_], got"
            msg += f" {list(exog_inflation.shape)} shape for exog_inflation."
            raise ValueError(msg)
    else:
        if exog_inflation.shape[1] != dim:
            msg = "Your formula inflation is {inflation_formula}."
            msg = f"exog_inflation should have shape [_,{dim}], got"
            msg += f" {list(exog_inflation.shape)} shape for exog_inflation."
            raise ValueError(msg)


def _pca_pairplot(array, n_components, dim, colors):
    """
    Generates a scatter matrix plot based on Principal Component Analysis (PCA) on
    the given array.

    Parameters
    ----------
    array: (np.ndarray): The array on which we will perform pca and then visualize.

    n_components (int, optional): The number of components to consider for plotting.
        If not specified, the maximum number of components will be used. Note that
        it will not display more than 10 graphs.
        Defaults to None.

    colors (np.ndarray): An array with one label for each
        sample in the endog property of the object.
        Defaults to None.
    Raises
    ------
    ValueError: If the number of components requested is greater than the number of variables in the dataset.
    """

    if n_components > dim:
        raise ValueError(
            f"You ask more components ({n_components}) than variables ({dim})"
        )
    if n_components > 10:
        msg = f"Can not display a scatter matrix with {n_components}*"
        msg += f"{n_components} = {n_components*n_components} graphs."
        msg += f" Setting the number of components to 10."
        warnings.warn(msg)
        n_components = 10
    pca = PCA(n_components=n_components)
    proj_variables = pca.fit_transform(array)
    components = torch.from_numpy(pca.components_)
    labels = {
        str(i): f"PC{i+1}: {np.round(pca.explained_variance_ratio_*100, 1)[i]}%"
        for i in range(n_components)
    }
    data = pd.DataFrame(proj_variables)
    data.columns = labels.values()
    if colors is not None:
        data["labels"] = colors
        fig = sns.pairplot(data, hue="labels")
    else:
        fig = sns.pairplot(data)
    plt.show()


def _check_full_rank_exog(exog, inflation=False):
    mat = exog.T @ exog
    d = mat.shape[1]
    rank = torch.linalg.matrix_rank(mat)
    if rank != d:
        print("shape::", exog)
        if inflation is True:
            name_mat = "exog_inflation"
            add_const_name = "add_const_inflation"
        else:
            name_mat = "exog"
            add_const_name = "add_const"
        msg = f"Input matrix {name_mat} does not result in {name_mat}.T @{name_mat} being full rank "
        msg += f"(rank = {rank}, expected = {d}). You may consider to remove one or more variables"
        msg += f" or set {add_const_name} to False if that is not already the case."
        msg += f" You can also set 0 + {name_mat} in the formula to avoid adding  "
        msg += "an intercept."
        raise ValueError(msg)


def threshold_samples_and_dim(max_samples, max_dim, n_samples, dim):
    if n_samples > max_samples:
        warnings.warn(
            f"\nTaking the whole max_samples samples of the dataset. Requested:n_samples={n_samples}, returned:{max_samples}"
        )
        n_samples = max_samples
    if dim > max_dim:
        warnings.warn(
            f"\nTaking the whole max_dim variables. Requested:dim={dim}, returned:{max_dim}"
        )
        dim = max_dim
    return n_samples, dim


def _check_right_exog_inflation_shape(
    exog_inflation, n_samples, dim, zero_inflation_formula
):
    if zero_inflation_formula == "global":
        if exog_inflation is not None:
            msg = "Can not set the exog_inflation to a value of the "
            msg += "zero inflation is 'global'."
            raise ValueError(msg)

    if zero_inflation_formula == "row-wise":
        if exog_inflation.shape[1] != dim:
            msg = f"Shape should be (_,{dim}), got shape{exog_inflation.shape}."
            raise ValueError(msg)
    if zero_inflation_formula == "column-wise":
        if exog_inflation.shape[0] != n_samples:
            msg = f"Shape should be ({n_samples},_), got shape{exog_inflation.shape}."
            raise ValueError(msg)


def mse(t):
    return torch.mean(t**2)
