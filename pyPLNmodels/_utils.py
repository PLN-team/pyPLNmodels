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
from typing import Optional, Dict, Any, Union, Tuple, List
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
            The size of the window for computing the criterion.
        """
        self.window = window
        self.running_times = []
        self.criterions = [1] * window  # the first window criterion won't be computed.
        self._elbos_list = []

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
            self.running_times[self.window :],
            self.criterions[self.window :],
            label="Delta",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Delta")
        ax.set_title("Increments")
        ax.legend()


def _sigmoid(tens: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-tens))


def sample_pln(pln_param, seed: int = None, return_latent=False) -> torch.Tensor:
    """
    Sample from the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    pln_param : PlnParameters object
        parameters of the model, containing the coeficient, the covariates,
        the components and the offsets.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    return_latent : bool, optional
        If True will return also the latent variables. Default is False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_latent is True
        Tuple containing counts (torch.Tensor), gaussian (torch.Tensor), and ksi (torch.Tensor)
    torch.Tensor if return_latent is False

    See also :func:`~pyPLNmodels.PlnParameters`
    """
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)

    n_samples = pln_param.offsets.shape[0]
    rank = pln_param.components.shape[1]

    if pln_param.covariates is None:
        XB = 0
    else:
        XB = torch.matmul(pln_param.covariates, pln_param.coef)

    gaussian = (
        torch.mm(torch.randn(n_samples, rank, device=DEVICE), pln_param.components.T)
        + XB
    )
    parameter = torch.exp(pln_param.offsets + gaussian)
    if pln_param.coef_inflation is not None:
        print("ZIPln is sampled")
        zero_inflated_mean = torch.matmul(
            pln_param.covariates, pln_param.coef_inflation
        )
        ksi = torch.bernoulli(1 / (1 + torch.exp(-zero_inflated_mean)))
    else:
        ksi = 0

    counts = (1 - ksi) * torch.poisson(parameter)

    torch.random.set_rng_state(prev_state)
    if return_latent is True:
        return counts, gaussian, ksi
    return counts


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


def _format_data(data: pd.DataFrame) -> torch.Tensor or None:
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
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    offsets_formula: str,
    take_log_offsets: bool,
    add_const: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Format each of the model parameters to an array or None if None.

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
    add_const: bool
        Whether to add a column of one in the covariates.
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
        raise ValueError("Counts should be only non negative values.")
    covariates = _format_data(covariates)
    if add_const is True:
        if covariates is None:
            covariates = torch.ones(counts.shape[0], 1)
        else:
            if _has_null_variance(covariates) is False:
                covariates = torch.concat(
                    (covariates, torch.ones(counts.shape[0]).unsqueeze(1)), dim=1
                )
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
    counts: torch.Tensor, covariates: torch.Tensor, offsets: torch.Tensor
) -> None:
    """
    Check if the shape of the input data is valid.

    Parameters
    ----------
    counts : torch.Tensor, shape (n, p)
        Count data.
    covariates : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, p) or None
        Offset data.
    """
    n_counts, p_counts = counts.shape
    n_offsets, p_offsets = offsets.shape
    _check_two_dimensions_are_equal("counts", "offsets", n_counts, n_offsets, 0, 0)
    if covariates is not None:
        n_cov, _ = covariates.shape
        _check_two_dimensions_are_equal("counts", "covariates", n_counts, n_cov, 0, 0)
    _check_two_dimensions_are_equal("counts", "offsets", p_counts, p_offsets, 1, 1)


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


def _get_simulation_coef_cov_offsets(
    n_samples: int, nb_cov: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get offsets, covariance coefficients with right shapes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    nb_cov : int
        Number of covariates. If 0, covariates will be None.
    dim : int
        Dimension required of the data.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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
            device="cpu",
        )
    coef = torch.randn(nb_cov, dim, device="cpu")
    offsets = torch.randint(
        low=0, high=2, size=(n_samples, dim), dtype=torch.float64, device="cpu"
    )
    torch.random.set_rng_state(prev_state)
    return coef, covariates, offsets


class PlnParameters:
    def __init__(self, components, coef, covariates, offsets, coef_inflation=None):
        """
        Instantiate all the needed parameters to sample from the PLN model.

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

        """
        self.components = _format_data(components)
        self.coef = _format_data(coef)
        self.covariates = _format_data(covariates)
        self.offsets = _format_data(offsets)
        self.coef_inflation = _format_data(coef_inflation)
        _check_two_dimensions_are_equal(
            "components", "coef", self.components.shape[0], self.coef.shape[1], 0, 1
        )
        if self.offsets is not None:
            _check_two_dimensions_are_equal(
                "components",
                "offsets",
                self.components.shape[0],
                self.offsets.shape[1],
                0,
                1,
            )
        if self.covariates is not None:
            _check_two_dimensions_are_equal(
                "offsets",
                "covariates",
                self.offsets.shape[0],
                self.covariates.shape[0],
                0,
                0,
            )
            _check_two_dimensions_are_equal(
                "covariates", "coef", self.covariates.shape[1], self.coef.shape[0], 1, 0
            )
        for array in [self.components, self.coef, self.covariates, self.offsets]:
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
        return self.components @ self.components.T


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
    n_samples: int = 100, dim: int = 25, nb_cov: int = 1, rank: int = 5
) -> PlnParameters:
    """
    Generate simulation parameters for a Poisson-lognormal model.

    Parameters
    ----------
        n_samples : int, optional
            The number of samples, by default 100.
        dim : int, optional
            The dimension of the data, by default 25.
        nb_cov : int, optional
            The number of covariates, by default 1.
        rank : int, optional
            The rank of the data components, by default 5.

    Returns
    -------
        PlnParameters
            The generated simulation parameters.

    """
    coef, covariates, offsets = _get_simulation_coef_cov_offsets(n_samples, nb_cov, dim)
    components = _get_simulation_components(dim, rank)
    return PlnParameters(components, coef, covariates, offsets)


def get_simulated_count_data(
    n_samples: int = 100,
    dim: int = 25,
    rank: int = 5,
    nb_cov: int = 1,
    return_true_param: bool = False,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulated count data from the PlnPCA model.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 100.
    dim : int, optional
        Dimension, by default 25.
    rank : int, optional
        Rank of the covariance matrix, by default 5.
    nb_cov : int, optional
        Number of covariates, by default 1.
    return_true_param : bool, optional
        Whether to return the true parameters of the model, by default False.
    seed : int, optional
        Seed value for random number generation, by default 0.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing counts, covariates, and offsets.
    """
    pln_param = get_simulation_parameters(n_samples, dim, nb_cov, rank)
    counts = sample_pln(pln_param, seed=seed, return_latent=False)
    if return_true_param is True:
        return (
            counts,
            pln_param.covariates,
            pln_param.offsets,
            pln_param.covariance,
            pln_param.coef,
        )
    return pln_param.counts, pln_param.cov, pln_param.offsets


def get_real_count_data(
    n_samples: int = 469, dim: int = 200, return_labels: bool = False
) -> np.ndarray:
    """
    Get real count data from the scMARK dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default max_samples.
    dim : int, optional
        Dimension, by default max_dim.
    return_labels: bool, optional
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
    counts_stream = pkg_resources.resource_stream(__name__, "data/scRT/counts.csv")
    counts = pd.read_csv(counts_stream).values[:n_samples, :dim]
    print(f"Returning dataset of size {counts.shape}")
    if return_labels is False:
        return counts
    labels_stream = pkg_resources.resource_stream(__name__, "data/scRT/labels.csv")
    labels = np.array(pd.read_csv(labels_stream).values[:n_samples].squeeze())
    return counts, labels


def load_model(path_of_directory: str) -> Dict[str, Any]:
    """
    Load model from the given directory for future initialization.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the model.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded model.

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
            except pd.errors.EmptyDataError:
                print(
                    f"Can't load {parameter} since empty. Standard initialization will be performed for this parameter"
                )
    os.chdir(working_dir)
    return data


def load_pln(path_of_directory: str) -> Dict[str, Any]:
    """
    Alias for :func:`~pyPLNmodels._utils.load_model`.
    """
    return load_model(path_of_directory)


def load_plnpcacollection(
    path_of_directory: str, ranks: Optional[List[int]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Load PlnPCAcollection models from the given directory.

    Parameters
    ----------
    path_of_directory : str
        The path to the directory containing the PlnPCAcollection models.
    ranks : List[int], optional
        A List of ranks specifying which models to load. If None, all models in the directory will be loaded.

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


def _extract_data_from_formula(formula: str, data: Dict[str, Any]) -> Tuple:
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
    obj: Union[np.ndarray, torch.Tensor, pd.DataFrame, None]
) -> Union[torch.Tensor, None]:
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
    counts,
    covariates,
    offsets,
    offsets_formula: str,
    take_log_offsets: bool,
    add_const: bool,
) -> tuple:
    """
    Handle the input data for the model.

    Parameters
    ----------
        counts : The counts data. If a DataFrame is provided, the column names are stored for later use.
        covariates : The covariates data.
        offsets : The offsets data.
        offsets_formula : The formula used for offsets.
        take_log_offsets : Indicates whether to take the logarithm of the offsets.
        add_const : Indicates whether to add a constant column to the covariates.

    Returns
    -------
        tuple: A tuple containing the processed counts, covariates, offsets, and column counts (if available).

    Raises
    ------
        ValueError: If the shapes of counts, covariates, and offsets do not match.
    """
    if isinstance(counts, pd.DataFrame):
        column_counts = counts.columns
    else:
        column_counts = None

    counts, covariates, offsets = _format_model_param(
        counts, covariates, offsets, offsets_formula, take_log_offsets, add_const
    )
    _check_data_shape(counts, covariates, offsets)
    return counts, covariates, offsets, column_counts
