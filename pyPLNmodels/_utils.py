import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.linalg as TLA
from matplotlib import transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from patsy import dmatrices
from typing import Optional, Dict, Any, Union
import pkg_resources


torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class _PlotArgs:
    def __init__(self, window: int):
        """
        Initialize the PlotArgs class.

        Parameters
        ----------
        window : int
            The size of the window for running statistics.
        """
        self.window = window
        self.running_times = []
        self.criterions = [1] * window
        self._elbos_list = []

    @property
    def iteration_number(self) -> int:
        """
        Get the number of iterations.

        Returns
        -------
        int
            The number of iterations.
        """
        return len(self._elbos_list)

    def _show_loss(self, ax=None, name_doss=""):
        """
        Show the loss plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, the current axes will be used.
        name_doss : str, optional
            The name of the loss. Default is an empty string.
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
        Show the stopping criterion plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on. If not provided, the current axes will be used.
        """
        ax = plt.gca() if ax is None else ax
        ax.plot(
            self.running_times[self.window :],
            self.criterions[self.window :],
            label="Delta",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Delta")
        ax.set_title("Increments")
        ax.legend()


def _init_covariance(
    counts: torch.Tensor, covariates: torch.Tensor, coef: torch.Tensor
) -> torch.Tensor:
    """
    Initialization for covariance for the Pln model. Take the log of counts
    (careful when counts=0), remove the covariates effects X@coef and
    then do as a MLE for Gaussians samples.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)

    Returns
    -------
    torch.Tensor
        Covariance matrix of size (p,p)
    """
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    log_y_centered = log_y - torch.mean(log_y, axis=0)
    n_samples = counts.shape[0]
    sigma_hat = 1 / (n_samples - 1) * (log_y_centered.T) @ log_y_centered
    return sigma_hat


def _init_components(
    counts: torch.Tensor, covariates: torch.Tensor, coef: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    Initialization for components for the Pln model. Get a first guess for covariance
    that is easier to estimate and then takes the rank largest eigenvectors to get components.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)
    rank : int
        The dimension of the latent space, i.e. the reduced dimension.

    Returns
    -------
    torch.Tensor
        Initialization of components of size (p,rank)
    """
    sigma_hat = _init_covariance(counts, covariates, coef).detach()
    components = _components_from_covariance(sigma_hat, rank)
    return components


def _init_latent_mean(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    coef: torch.Tensor,
    components: torch.Tensor,
    n_iter_max=500,
    lr=0.01,
    eps=7e-3,
) -> torch.Tensor:
    """
    Initialization for the variational parameter M. Basically, the mode of the log_posterior is computed.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)
    components : torch.Tensor
        Components of size (p,rank)
    n_iter_max : int, optional
        The maximum number of iterations in the gradient ascent. Default is 500.
    lr : float, optional
        The learning rate of the optimizer. Default is 0.01.
    eps : float, optional
        The tolerance. The algorithm will stop if the maximum of |W_t-W_{t-1}| is lower than eps,
        where W_t is the t-th iteration of the algorithm. Default is 7e-3.

    Returns
    -------
    torch.Tensor
        The initialized latent mean with size (n,rank)
    """
    mode = torch.randn(counts.shape[0], components.shape[1], device=DEVICE)
    mode.requires_grad_(True)
    optimizer = torch.optim.Rprop([mode], lr=lr)
    crit = 2 * eps
    old_mode = torch.clone(mode)
    keep_condition = True
    i = 0
    while i < n_iter_max and keep_condition:
        batch_loss = log_posterior(counts, covariates, offsets, mode, components, coef)
        loss = -torch.mean(batch_loss)
        loss.backward()
        optimizer.step()
        crit = torch.max(torch.abs(mode - old_mode))
        optimizer.zero_grad()
        if crit < eps and i > 2:
            keep_condition = False
        old_mode = torch.clone(mode)
        i += 1
    return mode


def _sigmoid(tens: torch.Tensor) -> torch.Tensor:
    """
    Compute the sigmoid function of x element-wise.

    Parameters
    ----------
    tens : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Output tensor with sigmoid applied element-wise
    """
    return 1 / (1 + torch.exp(-tens))


def sample_pln(
    components: torch.Tensor,
    coef: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    _coef_inflation: torch.Tensor = None,
    seed: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample from the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    components : torch.Tensor
        Components of size (p, rank)
    coef : torch.Tensor
        Coefficient of size (d, p)
    covariates : torch.Tensor or None
        Covariates, size (n, d) or None
    offsets : torch.Tensor
        Offset, size (n, p)
    _coef_inflation : torch.Tensor or None, optional
        Coefficient for zero-inflation model, size (d, p) or None. Default is None.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing counts (torch.Tensor), gaussian (torch.Tensor), and ksi (torch.Tensor)
    """
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)

    n_samples = offsets.shape[0]
    rank = components.shape[1]

    if covariates is None:
        XB = 0
    else:
        XB = torch.matmul(covariates, coef)

    gaussian = torch.mm(torch.randn(n_samples, rank, device=DEVICE), components.T) + XB
    parameter = torch.exp(offsets + gaussian)

    if _coef_inflation is not None:
        print("ZIPln is sampled")
        zero_inflated_mean = torch.matmul(covariates, _coef_inflation)
        ksi = torch.bernoulli(1 / (1 + torch.exp(-zero_inflated_mean)))
    else:
        ksi = 0

    counts = (1 - ksi) * torch.poisson(parameter)

    torch.random.set_rng_state(prev_state)
    return counts, gaussian, ksi


def _components_from_covariance(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Get the best matrix of size (p, rank) when covariance is of size (p, p),
    i.e., reduce norm(covariance - components @ components.T).

    Parameters
    ----------
    covariance : torch.Tensor
        Covariance matrix of size (p, p)
    rank : int
        The number of columns wanted for components

    Returns
    -------
    torch.Tensor
        Requested components of size (p, rank) containing the rank eigenvectors
        with largest eigenvalues.
    """
    eigenvalues, eigenvectors = TLA.eigh(covariance)
    requested_components = eigenvectors[:, -rank:] @ torch.diag(
        torch.sqrt(eigenvalues[-rank:])
    )
    return requested_components


def _init_coef(
    counts: torch.Tensor, covariates: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Initialize the coefficient for the Poisson regression model.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n, p)
    covariates : torch.Tensor
        Covariates, size (n, d)
    offsets : torch.Tensor
        Offset, size (n, p)

    Returns
    -------
    torch.Tensor or None
        Coefficient of size (d, p) or None if covariates is None.
    """
    if covariates is None:
        return None

    poiss_reg = _PoissonReg()
    poiss_reg.fit(counts, covariates, offsets)
    return poiss_reg.beta


def _log_stirling(integer: torch.Tensor) -> torch.Tensor:
    """
    Compute log(n!) even for large n using the Stirling formula to avoid numerical
    infinite values of n!.

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


def log_posterior(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    posterior_mean: torch.Tensor,
    components: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log posterior of the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (batch_size, p)
    covariates : torch.Tensor or None
        Covariates, size (batch_size, d) or (d)
    offsets : torch.Tensor
        Offset, size (batch_size, p)
    posterior_mean : torch.Tensor
        Posterior mean with size (N_samples, N_batch, rank) or (batch_size, rank)
    components : torch.Tensor
        Components with size (p, rank)
    coef : torch.Tensor
        Coefficient with size (d, p)

    Returns
    -------
    torch.Tensor
        Log posterior of size (N_samples, batch_size) or (batch_size).
    """
    length = len(posterior_mean.shape)
    rank = posterior_mean.shape[-1]
    components_posterior_mean = torch.matmul(
        components.unsqueeze(0), posterior_mean.unsqueeze(2)
    ).squeeze()

    if covariates is None:
        XB = 0
    else:
        XB = torch.matmul(covariates, coef)

    log_lambda = offsets + components_posterior_mean + XB
    first_term = (
        -rank / 2 * math.log(2 * math.pi)
        - 1 / 2 * torch.norm(posterior_mean, dim=-1) ** 2
    )
    second_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * counts - _log_stirling(counts), axis=-1
    )
    return first_term + second_term


def _trunc_log(tens: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """
    Compute the truncated logarithm of the input tensor.

    Parameters
    ----------
    tens : torch.Tensor
        Input tensor
    eps : float, optional
        Truncation value, default is 1e-16

    Returns
    -------
    torch.Tensor
        Truncated logarithm of the input tensor.
    """
    integer = torch.min(torch.max(tens, torch.tensor([eps])), torch.tensor([1 - eps]))
    return torch.log(integer)


def _get_offsets_from_sum_of_counts(counts: torch.Tensor) -> torch.Tensor:
    """
    Compute offsets from the sum of counts.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n, p)

    Returns
    -------
    torch.Tensor
        Offsets of size (n, p)
    """
    sum_of_counts = torch.sum(counts, axis=1)
    return sum_of_counts.repeat((counts.shape[1], 1)).T


def _raise_wrong_dimension_error(
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_of_error: int,
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
    dim_of_error : int
        Dimension causing the error

    Raises
    ------
    ValueError
        If the dimensions of the two tensors do not match at the non-singleton dimension.
    """
    msg = (
        f"The size of tensor {str_first_array} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_of_error}"
    )
    raise ValueError(msg)


def _check_two_dimensions_are_equal(
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_of_error: int,
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
    dim_of_error : int
        Dimension of the error.

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
            dim_of_error,
        )


def _init_S(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    beta: torch.Tensor,
    C: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """
    Initialize the S matrix.

    Parameters
    ----------
    counts : torch.Tensor, shape (n, )
        Count data.
    covariates : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, ) or None
        Offset data.
    beta : torch.Tensor, shape (d, )
        Beta parameter.
    C : torch.Tensor, shape (r, d)
        C parameter.
    M : torch.Tensor, shape (r, k)
        M parameter.

    Returns
    -------
    torch.Tensor, shape (r, r)
        Initialized S matrix.
    """
    n, rank = M.shape
    batch_matrix = torch.matmul(C[:, None, :], C[:, :, None])[None]
    CW = torch.matmul(C[None], M[:, None, :]).squeeze()
    common = torch.exp(offsets + covariates @ beta + CW)[:, None, None]
    prod = batch_matrix * common
    hess_posterior = torch.sum(prod, dim=1) + torch.eye(rank, device=DEVICE)
    inv_hess_posterior = -torch.inverse(hess_posterior)
    hess_posterior = torch.diagonal(inv_hess_posterior, dim1=-2, dim2=-1)
    return hess_posterior


def _format_data(data: pd.DataFrame) -> torch.Tensor or None:
    """
    Format the input data.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or torch.Tensor
        Input data.

    Returns
    -------
    torch.Tensor or None
        Formatted data.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).double().to(DEVICE)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).double().to(DEVICE)
    if isinstance(data, torch.Tensor):
        return data
    raise AttributeError(
        "Please insert either a numpy.ndarray, pandas.DataFrame or torch.Tensor"
    )


def _format_model_param(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    offsets_formula: str,
    take_log_offsets: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Format the model parameters.

    Parameters
    ----------
    counts : torch.Tensor or None, shape (n, )
        Count data.
    covariates : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, ) or None
        Offset data.
    offsets_formula : str
        Formula for calculating offsets.
    take_log_offsets : bool
        Flag indicating whether to take the logarithm of offsets.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Formatted model parameters.
    Raises
    ------
    ValueError
        If counts has negative values.

    """
    counts = _format_data(counts)
    if torch.min(counts) < 0:
        raise ValueError("Counts should be only non negavtive values.")
    if covariates is not None:
        covariates = _format_data(covariates)
    if offsets is None:
        if offsets_formula == "logsum":
            print("Setting the offsets as the log of the sum of counts")
            offsets = (
                torch.log(_get_offsets_from_sum_of_counts(counts)).double().to(DEVICE)
            )
        else:
            offsets = torch.zeros(counts.shape, device=DEVICE)
    else:
        offsets = _format_data(offsets).to(DEVICE)
        if take_log_offsets is True:
            offsets = torch.log(offsets)
    return counts, covariates, offsets


def _check_data_shape(
    counts: torch.Tensor, covariates: torch.Tensor, offsets: torch.Tensor
) -> None:
    """
    Check the shape of the input data.

    Parameters
    ----------
    counts : torch.Tensor, shape (n, p)
        Count data.
    covariates : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, p) or None
        Offset data.

    Raises
    ------
    ValueError
        If the dimensions of the input data do not match.
    """
    n_counts, p_counts = counts.shape
    n_offsets, p_offsets = offsets.shape
    _check_two_dimensions_are_equal("counts", "offsets", n_counts, n_offsets, 0)
    if covariates is not None:
        n_cov, _ = covariates.shape
        _check_two_dimensions_are_equal("counts", "covariates", n_counts, n_cov, 0)
    _check_two_dimensions_are_equal("counts", "offsets", p_counts, p_offsets, 1)


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
    Plot an ellipse on the given axes.

    Parameters:
    -----------
    mean_x : float
        Mean value of x-coordinate.
    mean_y : float
        Mean value of y-coordinate.
    cov : np.ndarray
        Covariance matrix.
    ax : object
        Axes object to plot the ellipse on.

    Returns:
    --------
    float
        Pearson correlation coefficient.
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
    return pearson


def _get_components_simulation(dim: int, rank: int) -> torch.Tensor:
    """
    Get the components for simulation.

    Parameters:
    -----------
    dim : int
        Dimension.
    rank : int
        Rank.

    Returns:
    --------
    torch.Tensor
        Components for simulation.
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
    return components.to(DEVICE)


def get_simulation_offsets_cov_coef(
    n_samples: int, nb_cov: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulation offsets, covariance coefficients.

    Parameters:
    -----------
    n_samples : int
        Number of samples.
    nb_cov : int
        Number of covariates.
    dim : int
        Dimension.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing offsets, covariates, and coefficients.
    """
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(0)
    if nb_cov == 0:
        covariates = None
    else:
        covariates = torch.randint(
            low=-1,
            high=2,
            size=(n_samples, nb_cov),
            dtype=torch.float64,
            device=DEVICE,
        )
    coef = torch.randn(nb_cov, dim, device=DEVICE)
    offsets = torch.randint(
        low=0, high=2, size=(n_samples, dim), dtype=torch.float64, device=DEVICE
    )
    torch.random.set_rng_state(prev_state)
    return offsets, covariates, coef


def get_simulated_count_data(
    n_samples: int = 100,
    dim: int = 25,
    rank: int = 5,
    nb_cov: int = 1,
    return_true_param: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulated count data.

    Parameters:
    -----------
    n_samples : int, optional
        Number of samples, by default 100.
    dim : int, optional
        Dimension, by default 25.
    rank : int, optional
        Rank, by default 5.
    nb_cov : int, optional
        Number of covariates, by default 1.
    return_true_param : bool, optional
        Whether to return true parameters, by default False.
    seed : int, optional
        Seed value for random number generation, by default 0.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing counts, covariates, and offsets.
    """
    components = _get_components_simulation(dim, rank)
    offsets, cov, true_coef = get_simulation_offsets_cov_coef(n_samples, nb_cov, dim)
    true_covariance = torch.matmul(components, components.T)
    counts, _, _ = sample_pln(components, true_coef, cov, offsets, seed=seed)
    if return_true_param is True:
        return counts, cov, offsets, true_covariance, true_coef
    return counts, cov, offsets


def get_real_count_data(n_samples: int = 270, dim: int = 100) -> np.ndarray:
    """
    Get real count data.

    Parameters:
    -----------
    n_samples : int, optional
        Number of samples, by default 270.
    dim : int, optional
        Dimension, by default 100.

    Returns:
    --------
    np.ndarray
        Real count data.
    """
    if n_samples > 297:
        warnings.warn(
            f"\nTaking the whole 270 samples of the dataset. Requested:n_samples={n_samples}, returned:270"
        )
        n_samples = 270
    if dim > 100:
        warnings.warn(
            f"\nTaking the whole 100 variables. Requested:dim={dim}, returned:100"
        )
        dim = 100
    counts_stream = pkg_resources.resource_stream(__name__, "data/scRT/Y_mark.csv")
    counts = pd.read_csv(counts_stream).values[:n_samples, :dim]
    # counts = pd.read_csv("./pyPLNmodels/data/scRT/Y_mark.csv").values[
    # :n_samples, :dim
    # ]
    print(f"Returning dataset of size {counts.shape}")
    return counts


def _closest(lst: list[float], element: float) -> float:
    """
    Find the closest element in a list to a given element.

    Parameters:
    -----------
    lst : list[float]
        List of float values.
    element : float
        Element to find the closest value to.

    Returns:
    --------
    float
        Closest element in the list.
    """
    lst = np.asarray(lst)
    idx = (np.abs(lst - element)).argmin()
    return lst[idx]


def load_model(path_of_directory: str) -> Dict[str, Any]:
    """
    Load models from the given directory.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the models.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded models.

    """
    working_dir = os.getcwd()
    os.chdir(path_of_directory)
    all_files = os.listdir()
    data = {}
    for filename in all_files:
        if filename.endswith(".csv"):
            parameter = filename[:-4]
            try:
                data[parameter] = pd.read_csv(filename, header=None).values
            except pd.errors.EmptyDataError as err:
                print(
                    f"Can't load {parameter} since empty. Standard initialization will be performed"
                )
    os.chdir(working_dir)
    return data


def load_pln(path_of_directory: str) -> Dict[str, Any]:
    """
    Load Pln models from the given directory.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the Pln models.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded Pln models.

    """
    return load_model(path_of_directory)


def load_plnpcacollection(
    path_of_directory: str, ranks: Optional[list[int]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Load PlnPCAcollection models from the given directory.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the PlnPCAcollection models.
    ranks : list[int], optional
        A list of ranks specifying which models to load. If None, all models in the directory will be loaded.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        A dictionary containing the loaded PlnPCAcollection models, with ranks as keys.

    Raises
    ------
    ValueError
        If an invalid model name is encountered and the rank cannot be determined.

    """
    working_dir = os.getcwd()
    os.chdir(path_of_directory)
    if ranks is None:
        dirnames = os.listdir()
        ranks = []
        for dirname in dirnames:
            try:
                rank = int(dirname[-1])
            except ValueError:
                raise ValueError(
                    f"Can't load the model {dirname}. End of {dirname} should be an int"
                )
            ranks.append(rank)
    datas = {}
    for rank in ranks:
        datas[rank] = load_model(f"PlnPCA_rank_{rank}")
    os.chdir(working_dir)
    return datas


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


def _extract_data_from_formula(formula: str, data: Dict[str, Any]) -> tuple:
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
    tuple
        A tuple containing the extracted counts, covariates, and offsets.

    """
    dmatrix = dmatrices(formula, data=data)
    counts = dmatrix[0]
    covariates = dmatrix[1]
    if covariates.size == 0:
        covariates = None
    offsets = data.get("offsets", None)
    return counts, covariates, offsets


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
    return isinstance(dictionary[list(dictionary.keys())[0]], dict)


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
    obj: Union[np.ndarray, torch.Tensor, pd.DataFrame, None]
) -> Union[torch.Tensor, None]:
    """
    Convert an object to a PyTorch tensor.

    Parameters:
    ----------
        obj (np.ndarray or torch.Tensor or pd.DataFrame or None):
            The object to be converted.

    Returns:
        torch.Tensor or None:
            The converted PyTorch tensor.

    Raises:
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


class _PoissonReg:
    """
    Poisson regression model.

    Attributes
    ----------
    beta : torch.Tensor
        The learned regression coefficients.

    Methods
    -------
    fit(Y, covariates, O, Niter_max=300, tol=0.001, lr=0.005, verbose=False)
        Fit the Poisson regression model to the given data.

    """

    def __init__(self) -> None:
        self.beta: Optional[torch.Tensor] = None

    def fit(
        self,
        Y: torch.Tensor,
        covariates: torch.Tensor,
        O: torch.Tensor,
        Niter_max: int = 300,
        tol: float = 0.001,
        lr: float = 0.005,
        verbose: bool = False,
    ) -> None:
        """
        Fit the Poisson regression model to the given data.

        Parameters
        ----------
        Y : torch.Tensor
            The dependent variable of shape (n_samples, n_features).
        covariates : torch.Tensor
            The covariates of shape (n_samples, n_covariates).
        O : torch.Tensor
            The offset term of shape (n_samples, n_features).
        Niter_max : int, optional
            The maximum number of iterations (default is 300).
        tol : float, optional
            The tolerance for convergence (default is 0.001).
        lr : float, optional
            The learning rate (default is 0.005).
        verbose : bool, optional
            Whether to print intermediate information during fitting (default is False).

        """
        beta = torch.rand(
            (covariates.shape[1], Y.shape[1]), device=DEVICE, requires_grad=True
        )
        optimizer = torch.optim.Rprop([beta], lr=lr)
        i = 0
        grad_norm = 2 * tol  # Criterion
        while i < Niter_max and grad_norm > tol:
            loss = -compute_poissreg_log_like(Y, O, covariates, beta)
            loss.backward()
            optimizer.step()
            grad_norm = torch.norm(beta.grad)
            beta.grad.zero_()
            i += 1
            if verbose:
                if i % 10 == 0:
                    print("log like : ", -loss)
                    print("grad_norm : ", grad_norm)
                if i < Niter_max:
                    print("Tolerance reached in {} iterations".format(i))
                else:
                    print("Maximum number of iterations reached")
        self.beta = beta


def compute_poissreg_log_like(
    Y: torch.Tensor, O: torch.Tensor, covariates: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log likelihood of a Poisson regression model.

    Parameters
    ----------
    Y : torch.Tensor
        The dependent variable of shape (n_samples, n_features).
    O : torch.Tensor
        The offset term of shape (n_samples, n_features).
    covariates : torch.Tensor
        The covariates of shape (n_samples, n_covariates).
    beta : torch.Tensor
        The regression coefficients of shape (n_covariates, n_features).

    Returns
    -------
    torch.Tensor
        The log likelihood of the Poisson regression model.

    """
    XB = torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    return torch.sum(-torch.exp(O + XB) + torch.multiply(Y, O + XB))


def array2tensor(func):
    def setter(self, array_like):
        array_like = _to_tensor(array_like)
        func(self, array_like)

    return setter
