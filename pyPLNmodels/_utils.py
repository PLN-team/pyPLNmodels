import os
import math
import warnings
import textwrap
from typing import Optional, Dict, Any, Union, Tuple, List
import pkg_resources

import numpy as np
import pandas as pd
import torch
import torch.linalg as TLA
from matplotlib import transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Circle


torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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

    def update_criterion(self, elbo, running_time):
        self._elbos_list.append(elbo)
        self.running_times.append(running_time)
        self.cumulative_elbo_list.append(self.cumulative_elbo + elbo)
        self.normalized_elbo_list.append(-elbo / self.cumulative_elbo_list[-1])
        if self.iteration_number > 1:
            current_derivative = np.abs(
                (self.normalized_elbo_list[-2] - self.normalized_elbo_list[-1])
                / (self.running_times[-2] - self.running_times[-1])
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


def sample_pln(pln_param, *, seed: int = None, return_latent=False) -> torch.Tensor:
    """
    Sample from the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    pln_param : PlnParameters object
        parameters of the model, containing the coeficient, the exog,
        the components and the offsets.
    seed : int or None, optional(keyword-only)
        Random seed for reproducibility. Default is None.
    return_latent : bool, optional(keyword-only)
        If True will return also the latent variables. Default is False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_latent is True
        Tuple containing endog (torch.Tensor), gaussian (torch.Tensor), and ksi (torch.Tensor)
    torch.Tensor if return_latent is False

    See also :func:`~pyPLNmodels.PlnParameters`
    """
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)

    n_samples = pln_param.offsets.shape[0]
    rank = pln_param.components.shape[1]

    if pln_param.exog is None:
        XB = 0
    else:
        XB = torch.matmul(pln_param.exog, pln_param.coef)

    gaussian = (
        torch.mm(torch.randn(n_samples, rank, device=DEVICE), pln_param.components.T)
        + XB
    )
    parameter = torch.exp(pln_param.offsets + gaussian)
    if pln_param.coef_inflation is not None:
        print("ZIPln is sampled")
        zero_inflated_mean = torch.matmul(pln_param.exog, pln_param.coef_inflation)
        ksi = torch.bernoulli(1 / (1 + torch.exp(-zero_inflated_mean)))
    else:
        ksi = 0

    endog = (1 - ksi) * torch.poisson(parameter)

    torch.random.set_rng_state(prev_state)
    if return_latent is True:
        return endog, gaussian, ksi
    return endog


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


def _trunc_log(tens: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    integer = torch.min(
        torch.max(tens, torch.tensor([eps]).to(DEVICE)),
        torch.tensor([1 - eps]).to(DEVICE),
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
        return torch.from_numpy(data.values).double().to(DEVICE)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).double().to(DEVICE)
    if isinstance(data, torch.Tensor):
        return data.to(DEVICE)
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
    if torch.min(torch.sum(endog, axis=1)) < 0.5:
        raise ValueError(
            "Counts contains individuals containing only zero counts. Remove it."
        )
    exog = _format_data(exog)
    if add_const is True:
        if exog is None:
            exog = torch.ones(endog.shape[0], 1).to(DEVICE)
        else:
            if _has_null_variance(exog) is False:
                exog = torch.concat(
                    (exog, torch.ones(endog.shape[0]).unsqueeze(1)), dim=1
                ).to(DEVICE)
    if offsets is None:
        if offsets_formula == "logsum":
            print("Setting the offsets as the log of the sum of endog")
            offsets = (
                torch.log(_get_offsets_from_sum_of_endog(endog)).double().to(DEVICE)
            )
        elif offsets_formula == "zero":
            print("Setting the offsets to zero")
            offsets = torch.zeros(endog.shape, device=DEVICE)
        else:
            raise ValueError(
                'Wrong offsets_formula. Expected either "zero" or "logsum", got {offsets_formula}'
            )
    else:
        offsets = _format_data(offsets).to(DEVICE)
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
    _check_two_dimensions_are_equal("endog", "offsets", n_endog, n_offsets, 0, 0)
    if exog is not None:
        n_cov, _ = exog.shape
        _check_two_dimensions_are_equal("endog", "exog", n_endog, n_cov, 0, 0)
    _check_two_dimensions_are_equal("endog", "offsets", p_endog, p_offsets, 1, 1)


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


def _get_simulation_components(dim: int, rank: int) -> torch.Tensor:
    """
    Get the components for simulation. The resulting covariance matrix
    will be a matrix per blocks plus a little noise.

    Parameters
    ----------
    dim : int
        Dimension of the data.
    rank : int
        Rank of the resulting covariance matrix (i.e. number of components).

    Returns
    -------
    torch.Tensor
        Components.
    """
    block_size = dim // rank
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(0)
    components = torch.zeros(dim, rank)
    for column_number in range(rank):
        components[
            column_number * block_size : (column_number + 1) * block_size, column_number
        ] = 1
    components += torch.randn(dim, rank) / 8
    torch.random.set_rng_state(prev_state)
    return components.to("cpu")


def _get_simulation_coef_cov_offsets_coefzi(
    n_samples: int,
    nb_cov: int,
    dim: int,
    add_const: bool,
    zero_inflated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get offsets, covariance coefficients with right shapes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    nb_cov : int
        Number of exog. If 0, exog will be None,
        unless add_const is True.
        If add_const is True, then there will be nb_cov+1
        exog as the intercept can be seen as a exog.
    dim : int
        Dimension required of the data.
    add_const : bool, optional
        If True, will add a vector of ones in the exog.
    zero_inflated : bool
        If True, will return a zero_inflated coefficient.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing offsets, exog, and coefficients.
    """
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(0)
    if nb_cov == 0:
        if add_const is True:
            exog = torch.ones(n_samples, 1)
        else:
            exog = None
    else:
        exog = torch.randint(
            low=-1,
            high=2,
            size=(n_samples, nb_cov),
            dtype=torch.float64,
            device="cpu",
        )
        if add_const is True:
            exog = torch.cat((exog, torch.ones(n_samples, 1)), axis=1)
    if exog is None:
        if zero_inflated is True:
            msg = "Can not instantiate a zero inflate model without covariates."
            msg += " Please give at least an intercept by setting add_const to True"
            raise ValueError(msg)
        coef = None
        coef_inflation = None
    else:
        coef = torch.randn(exog.shape[1], dim, device="cpu")
        if zero_inflated is True:
            coef_inflation = torch.randn(exog.shape[1], dim, device="cpu")
        else:
            coef_inflation = None
    offsets = torch.randint(
        low=0, high=2, size=(n_samples, dim), dtype=torch.float64, device="cpu"
    )
    torch.random.set_rng_state(prev_state)
    return coef, exog, offsets, coef_inflation


class PlnParameters:
    def __init__(
        self,
        *,
        components: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        coef: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        exog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        coef_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame, None] = None,
    ):
        """
        Instantiate all the needed parameters to sample from the PLN model.

        Parameters
        ----------
        components : : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Components of size (p, rank)
        coef : : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Coefficient of size (d, p)
        exog : : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None(keyword-only)
            Covariates, size (n, d) or None
        offsets : : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Offset, size (n, p)
        coef_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame, None], optional(keyword-only)
            Coefficient for zero-inflation model, size (d, p) or None. Default is None.
        """
        self._components = _format_data(components)
        self._coef = _format_data(coef)
        self._exog = _format_data(exog)
        self._offsets = _format_data(offsets)
        self._coef_inflation = _format_data(coef_inflation)
        if self._coef is not None:
            _check_two_dimensions_are_equal(
                "components",
                "coef",
                self._components.shape[0],
                self._coef.shape[1],
                0,
                1,
            )
        if self._offsets is not None:
            _check_two_dimensions_are_equal(
                "components",
                "offsets",
                self._components.shape[0],
                self._offsets.shape[1],
                0,
                1,
            )
        if self._exog is not None:
            _check_two_dimensions_are_equal(
                "offsets",
                "exog",
                self._offsets.shape[0],
                self._exog.shape[0],
                0,
                0,
            )
            _check_two_dimensions_are_equal(
                "exog",
                "coef",
                self._exog.shape[1],
                self._coef.shape[0],
                1,
                0,
            )
        for array in [self._components, self._coef, self._exog, self._offsets]:
            if array is not None:
                if len(array.shape) != 2:
                    raise RuntimeError(
                        f"Expected all arrays to be 2-dimensional, got {len(array.shape)}"
                    )

    @property
    def covariance(self):
        """
        Covariance of the model.
        """
        return self._components @ self._components.T

    @property
    def components(self):
        """
        Components of the model.
        """
        return self._components

    @property
    def offsets(self):
        """
        Data offsets.
        """
        return self._offsets

    @property
    def coef(self):
        """
        Coef of the model.
        """
        return self._coef

    @property
    def exog(self):
        """
        Data exog.
        """
        return self._exog

    @property
    def coef_inflation(self):
        """
        Inflation coefficient of the model.
        """
        return self._coef_inflation


def _check_two_dimensions_are_equal(
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


def get_simulation_parameters(
    *,
    n_samples: int = 100,
    dim: int = 25,
    nb_cov: int = 1,
    rank: int = 5,
    add_const: bool = True,
    zero_inflated: bool = False,
) -> PlnParameters:
    """
    Generate simulation parameters for a Poisson-lognormal model.

    Parameters
    ----------
        n_samples : int, optional(keyword-only)
            The number of samples, by default 100.
        dim : int, optional(keyword-only)
            The dimension of the data, by default 25.
        nb_cov : int, optional(keyword-only)
            The number of exog, by default 1. If add_const is True,
            then there will be nb_cov+1 exog as the intercept can be seen
            as a exog.
        rank : int, optional(keyword-only)
            The rank of the data components, by default 5.
        add_const : bool, optional(keyword-only)
            If True, will add a vector of ones in the exog.
        zero_inflated : bool, optional(keyword-only)
            If True, the model will be zero inflated.
            Default is False.

    Returns
    -------
        PlnParameters
            The generated simulation parameters.
    """
    coef, exog, offsets, coef_inflation = _get_simulation_coef_cov_offsets_coefzi(
        n_samples, nb_cov, dim, add_const, zero_inflated
    )
    components = _get_simulation_components(dim, rank)
    return PlnParameters(
        components=components,
        coef=coef,
        exog=exog,
        offsets=offsets,
        coef_inflation=coef_inflation,
    )


def get_simulated_count_data(
    *,
    n_samples: int = 100,
    dim: int = 25,
    rank: int = 5,
    nb_cov: int = 1,
    return_true_param: bool = False,
    add_const: bool = True,
    zero_inflated=False,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulated count data from the PlnPCA model.

    Parameters
    ----------
    n_samples : int, optional(keyword-only)
        Number of samples, by default 100.
    dim : int, optional(keyword-only)
        Dimension, by default 25.
    rank : int, optional(keyword-only)
        Rank of the covariance matrix, by default 5.
    add_const : bool, optional(keyword-only)
        If True, will add a vector of ones. Default is True
    nb_cov : int, optional(keyword-only)
        Number of exog, by default 1.
    return_true_param : bool, optional(keyword-only)
        Whether to return the true parameters of the model, by default False.
    zero_inflated: bool, optional(keyword-only)
        Whether to use a zero inflated model or not.
        Default to False.
    seed : int, optional(keyword-only)
        Seed value for random number generation, by default 0.

    Returns
    -------
    if return_true_param is False:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing endog, exog, and offsets.
    else:
        if zero_inflated is True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                Tuple containing endog, exog, offsets, covariance, coef, coef_inflation .
        else:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                Tuple containing endog, exog, offsets, covariance, coef.

    """
    pln_param = get_simulation_parameters(
        n_samples=n_samples,
        dim=dim,
        nb_cov=nb_cov,
        rank=rank,
        add_const=add_const,
        zero_inflated=zero_inflated,
    )
    endog = sample_pln(pln_param, seed=seed, return_latent=False)
    if return_true_param is True:
        if zero_inflated is True:
            return (
                endog,
                pln_param.exog,
                pln_param.offsets,
                pln_param.covariance,
                pln_param.coef,
                pln_param.coef_inflation,
            )
        return (
            endog,
            pln_param.exog,
            pln_param.offsets,
            pln_param.covariance,
            pln_param.coef,
        )
    return endog, pln_param.exog, pln_param.offsets


def get_real_count_data(
    *, n_samples: int = 469, dim: int = 200, return_labels: bool = False
) -> np.ndarray:
    """
    Get real count data from the scMARK dataset.

    Parameters
    ----------
    n_samples : int, optional(keyword-only)
        Number of samples, by default max_samples.
    dim : int, optional(keyword-only)
        Dimension, by default max_dim.
    return_labels: bool, optional(keyword-only)
        If True, will return the labels of the count data
    Returns
    -------
    np.ndarray
        Real count data and labels if return_labels is True.
    """
    max_samples = 469
    max_dim = 200
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
    endog_stream = pkg_resources.resource_stream(__name__, "data/scRT/counts.csv")
    endog = pd.read_csv(endog_stream).values[:n_samples, :dim]
    print(f"Returning dataset of size {endog.shape}")
    if return_labels is False:
        return endog
    labels_stream = pkg_resources.resource_stream(__name__, "data/scRT/labels.csv")
    labels = np.array(pd.read_csv(labels_stream).values[:n_samples].squeeze())
    return endog, labels


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


def _extract_data_from_formula(
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
    # dmatrices can not deal with GPU matrices
    for key, matrix in data.items():
        if isinstance(matrix, torch.Tensor):
            data[key] = matrix.cpu()
    dmatrix = dmatrices(formula, data=data)
    endog = dmatrix[0]
    exog = dmatrix[1]
    if exog.size == 0:
        exog = None
    offsets = data.get("offsets", None)
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
        return torch.from_numpy(obj).to(DEVICE)
    if isinstance(obj, torch.Tensor):
        return obj.to(DEVICE)
    if isinstance(obj, pd.DataFrame):
        return torch.from_numpy(obj.values).to(DEVICE)
    raise TypeError(
        "Please give either an np.ndarray or torch.Tensor or pd.DataFrame or None"
    )


def _array2tensor(func):
    def setter(self, array_like):
        array_like = _to_tensor(array_like)
        func(self, array_like)

    return setter


def _handle_data(
    endog,
    exog,
    offsets,
    offsets_formula: str,
    take_log_offsets: bool,
    add_const: bool,
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

    Returns
    -------
        tuple: A tuple containing the processed endog, exog, offsets, and column endog (if available).

    Raises
    ------
        ValueError: If the shapes of endog, exog, and offsets do not match.
    """
    if isinstance(endog, pd.DataFrame):
        column_endog = endog.columns
    else:
        column_endog = None

    endog, exog, offsets = _format_model_param(
        endog, exog, offsets, offsets_formula, take_log_offsets, add_const
    )
    _check_data_shape(endog, exog, offsets)
    return endog, exog, offsets, column_endog


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
    lamby = lambert(y)
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
    mask += t < -10
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
    print("TEST")
    print(sorted(plt.style.available))

    with plt.style.context(("seaborn-v0_8-whitegrid")):
        fig, axs = plt.subplots(figsize=(6, 6))
        plot_correlation_arrows(axs, ccircle, variables_names)

        # Draw the unit circle, for clarity
        circle = Circle(
            (0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5
        )
        axs.add_patch(circle)
        axs.set_xlabel(f"PCA 1 {(np.round(explained_ratio[0], 3))}")
        axs.set_ylabel(f"PCA 2 {(np.round(explained_ratio[1], 3))}")
        axs.set_title(f"Correlation circle on the transformed variables{title}")

    plt.tight_layout()
    plt.show()
