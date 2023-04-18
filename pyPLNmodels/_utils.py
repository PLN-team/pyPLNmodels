import math  # pylint:disable=[C0114]
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg as TLA
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import transforms

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class PLNPlotArgs:
    def __init__(self, window):
        self.window = window
        self.running_times = []
        self.criterions = [1] * window
        self.elbos_list = []

    @property
    def iteration_number(self):
        return len(self.elbos_list)

    def show_loss(self, ax=None, savefig=False, name_doss=""):
        """Show the ELBO of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The ELBO will be displayed in this ax
                if not None. If None, will simply create an axis. Default
                is None.
            'name_file': str. The name of the file the graphic
                will be saved to.
                Default is 'fastPLNPCA_ELBO'.
        returns: None but displays the ELBO.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(
            self.running_times,
            -np.array(self.elbos_list),
            label="Negative ELBO",
        )
        last_elbos = np.round(self.elbos_list[-1], 6)
        ax.set_title(f"Negative ELBO. Best ELBO ={last_elbos}")
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("ELBO")
        ax.legend()
        # save the graphic if needed
        if savefig:
            plt.savefig(name_doss)

    def show_stopping_criterion(self, ax=None, savefig=False, name_doss=""):
        """Show the criterion of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The criterion will be displayed
                in this ax
                if not None. If None, will simply create an axis.
                Default is None.
            'name_file': str. The name of the file the graphic will
                be saved to.
                Default is 'fastPLN_criterion'.
        returns: None but displays the criterion.
        """
        if ax is None:
            ax = plt.gca()
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
        # save the graphic if needed
        if savefig:
            plt.savefig(name_doss)


class PlnData:
    def __init__(self, counts, covariates, offsets):
        self._counts = counts
        self._covariates = covariates
        self._offsets = offsets


def init_sigma(counts, covariates, coef):
    """Initialization for covariance for the PLN model. Take the log of counts
    (careful when counts=0), remove the covariates effects X@coef and
    then do as a MLE for Gaussians samples.
    Args :
            counts: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            coef: torch.tensor of size (d,p)
    Returns : torch.tensor of size (p,p).
    """
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    log_y_centered = (
        log_y - torch.matmul(covariates.unsqueeze(1), coef.unsqueeze(0)).squeeze()
    )
    # MLE in a Gaussian setting
    n_samples = counts.shape[0]
    sigma_hat = 1 / (n_samples - 1) * (log_y_centered.T) @ log_y_centered
    return sigma_hat


def init_components(counts, covariates, coef, rank):
    """Inititalization for components for the PLN model. Get a first
    guess for covariance that is easier to estimate and then takes
    the rank largest eigenvectors to get components.
    Args :
        counts: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covarites: torch.tensor. Covariates, size (n,d)
        coef: torch.tensor of size (d,p)
        rank: int. The dimension of the latent space, i.e. the reducted dimension.
    Returns :
        torch.tensor of size (p,rank). The initialization of components.
    """
    sigma_hat = init_sigma(counts, covariates, coef).detach()
    components = components_from_covariance(sigma_hat, rank)
    return components


def init_latent_mean(
    counts, covariates, offsets, coef, components, n_iter_max=500, lr=0.01, eps=7e-3
):
    """Initialization for the variational parameter M. Basically,
    the mode of the log_posterior is computed.

    Args:
        counts: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        coef: torch.tensor of size (d,p)
        N_iter_max: int. The maximum number of iteration in
            the gradient ascent.
        lr: positive float. The learning rate of the optimizer.
        eps: positive float, optional. The tolerance. The algorithm will stop
            if the maximum of |W_t-W_{t-1}| is lower than eps, where W_t
            is the t-th iteration of the algorithm.This parameter
            changes a lot the resulting time of the algorithm. Default is 9e-3.
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


def sigmoid(tens):
    """Compute the sigmoid function of x element-wise."""
    return 1 / (1 + torch.exp(-tens))


def sample_pln(components, coef, covariates, offsets, _coef_inflation=None, seed=None):
    """Sample Poisson log Normal variables. If _coef_inflation is not None, the model will
    be zero inflated.

    Args:
        components: torch.tensor of size (p,rank). The matrix components of the PLN model
        coef: torch.tensor of size (d,p). Regression parameter.
        0: torch.tensor of size (n,p). Offsets.
        covariates : torch.tensor of size (n,d). Covariates.
        _coef_inflation: torch.tensor of size (d,p), optional. If _coef_inflation is not None,
             the ZIPLN model is chosen, so that it will add a
             Bernouilli layer. Default is None.
    Returns :
        counts: torch.tensor of size (n,p), the count variables.
        Z: torch.tensor of size (n,p), the gaussian latent variables.
        ksi: torch.tensor of size (n,p), the bernoulli latent variables
        (full of zeros if _coef_inflation is None).
    """
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    n_samples = offsets.shape[0]
    rank = components.shape[1]
    full_of_ones = torch.ones((n_samples, 1))
    if covariates is None:
        covariates = full_of_ones
    else:
        covariates = torch.stack((full_of_ones, covariates), axis=1).squeeze()
    gaussian = (
        torch.mm(torch.randn(n_samples, rank, device=DEVICE), components.T)
        + covariates @ coef
    )
    parameter = torch.exp(offsets + gaussian)
    if _coef_inflation is not None:
        print("ZIPLN is sampled")
        zero_inflated_mean = covariates @ _coef_inflation
        ksi = torch.bernoulli(1 / (1 + torch.exp(-zero_inflated_mean)))
    else:
        ksi = 0
    counts = (1 - ksi) * torch.poisson(parameter)
    torch.random.set_rng_state(prev_state)
    return counts, gaussian, ksi


# def logit(tens):
#     """logit function. If x is too close from 1, we set the result to 0.
#     performs logit element wise."""
#     return torch.nan_to_num(torch.log(x / (1 - tens)),
# nan=0, neginf=0, posinf=0)


def components_from_covariance(covariance, rank):
    """Get the best matrix of size (p,rank) when covariance is of
    size (p,p). i.e. reduces norm(covariance-components@components.T)
    Args :
        covariance: torch.tensor of size (p,p). Should be positive definite and
            symmetric.
        rank: int. The number of columns wanted for components

    Returns:
        components_reduct: torch.tensor of size (p,rank) containing the rank eigenvectors with
        largest eigenvalues.
    """
    eigenvalues, eigenvectors = TLA.eigh(covariance)
    requested_components = eigenvectors[:, -rank:] @ torch.diag(
        torch.sqrt(eigenvalues[-rank:])
    )
    return requested_components


def init_coef(counts, covariates):
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    log_y = log_y.to(DEVICE)
    return torch.matmul(
        torch.inverse(torch.matmul(covariates.T, covariates)),
        torch.matmul(covariates.T, log_y),
    )


def log_stirling(integer):
    """Compute log(n!) even for n large. We use the Stirling formula to avoid
    numerical infinite values of n!.
    Args:
         n: torch.tensor of any size.
    Returns:
        An approximation of log(n_!) element-wise.
    """
    integer_ = integer + (
        integer == 0
    )  # Replace the 0 with 1. It doesn't change anything since 0! = 1!
    return torch.log(torch.sqrt(2 * np.pi * integer_)) + integer_ * torch.log(
        integer_ / math.exp(1)
    )


def log_posterior(counts, covariates, offsets, posterior_mean, components, coef):
    """Compute the log posterior of the PLN model. Compute it either
    for posterior_mean of size (N_samples, N_batch,rank) or (batch_size, rank). Need to have
    both cases since it is done for both cases after. Please the mathematical
    description of the package for the formula.
    Args :
        counts : torch.tensor of size (batch_size, p)
        covariates : torch.tensor of size (batch_size, d) or (d)
    Returns: torch.tensor of size (N_samples, batch_size) or (batch_size).
    """
    length = len(posterior_mean.shape)
    rank = posterior_mean.shape[-1]
    if length == 2:
        components_posterior_mean = torch.matmul(
            components.unsqueeze(0), posterior_mean.unsqueeze(2)
        ).squeeze()
    elif length == 3:
        components_posterior_mean = torch.matmul(
            components.unsqueeze(0).unsqueeze(1), posterior_mean.unsqueeze(3)
        ).squeeze()

    log_lambda = offsets + components_posterior_mean + covariates @ coef
    first_term = (
        -rank / 2 * math.log(2 * math.pi)
        - 1 / 2 * torch.norm(posterior_mean, dim=-1) ** 2
    )
    second_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * counts - log_stirling(counts), axis=-1
    )
    return first_term + second_term


def trunc_log(tens, eps=1e-16):
    integer = torch.min(torch.max(tens, torch.tensor([eps])), torch.tensor([1 - eps]))
    return torch.log(integer)


def get_offsets_from_sum_of_counts(counts):
    sum_of_counts = torch.sum(counts, axis=1)
    return sum_of_counts.repeat((counts.shape[1], 1)).T


def raise_wrong_dimension_error(
    str_first_array, str_second_array, dim_first_array, dim_second_array, dim_of_error
):
    msg = (
        f"The size of tensor {str_first_array} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_of_error}"
    )
    raise ValueError(msg)


def check_two_dimensions_are_equal(
    str_first_array, str_second_array, dim_first_array, dim_second_array, dim_of_error
):
    if dim_first_array != dim_second_array:
        raise_wrong_dimension_error(
            str_first_array,
            str_second_array,
            dim_first_array,
            dim_second_array,
            dim_of_error,
        )


def format_data(data):
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).double().to(DEVICE)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).double().to(DEVICE)
    if isinstance(data, torch.Tensor):
        return data
    raise AttributeError(
        "Please insert either a numpy.ndarray, pandas.DataFrame or torch.Tensor"
    )


def format_model_param(counts, covariates, offsets, offsets_formula):
    counts = format_data(counts)
    covariates = prepare_covariates(covariates, counts.shape[0])
    if offsets is None:
        if offsets_formula == "logsum":
            print("Setting the offsets as the log of the sum of counts")
            offsets = (
                torch.log(get_offsets_from_sum_of_counts(counts)).double().to(DEVICE)
            )
        else:
            offsets = torch.zeros(counts.shape, device=DEVICE)
    else:
        offsets = format_data(offsets).to(DEVICE)
    return counts, covariates, offsets


def prepare_covariates(covariates, n_samples):
    full_of_ones = torch.full((n_samples, 1), 1, device=DEVICE).double()
    if covariates is None:
        return full_of_ones
    covariates = format_data(covariates)
    return torch.stack((full_of_ones, covariates), axis=1).squeeze()


def check_data_shape(counts, covariates, offsets):
    n_counts, p_counts = counts.shape
    n_offsets, p_offsets = offsets.shape
    n_cov, _ = covariates.shape
    check_two_dimensions_are_equal("counts", "offsets", n_counts, n_offsets, 0)
    check_two_dimensions_are_equal("counts", "covariates", n_counts, n_cov, 0)
    check_two_dimensions_are_equal("counts", "offsets", p_counts, p_offsets, 1)


def extract_cov_offsets_offsetsformula(dictionnary):
    covariates = dictionnary.get("covariates", None)
    offsets = dictionnary.get("offsets", None)
    offsets_formula = dictionnary.get("offsets_formula", None)
    return covariates, offsets, offsets_formula


def nice_string_of_dict(dictionnary):
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            return_string += f"{str(element):>12}"
        return_string += "\n"
    return return_string


def plot_ellipse(mean_x, mean_y, cov, ax):
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


def get_components_simulation(dim, rank):
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


def get_simulation_offsets_cov_coef(n_samples, nb_cov, dim):
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(0)
    if nb_cov < 2:
        covariates = None
    else:
        covariates = torch.randint(
            low=-1,
            high=2,
            size=(n_samples, nb_cov - 1),
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
    n_samples=100, dim=25, rank=5, nb_cov=1, return_true_param=False, seed=0
):
    components = get_components_simulation(dim, rank)
    offsets, cov, true_coef = get_simulation_offsets_cov_coef(n_samples, nb_cov, dim)
    true_covariance = torch.matmul(components, components.T)
    counts, _, _ = sample_pln(components, true_coef, cov, offsets, seed=seed)
    if return_true_param is True:
        return counts, cov, offsets, true_covariance, true_coef
    return counts, cov, offsets


def get_real_count_data(n_samples=270, dim=100):
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
    counts = pd.read_csv("../example_data/real_data/Y_mark.csv").values[
        :n_samples, :dim
    ]
    print(f"Returning dataset of size {counts.shape}")
    return counts


def closest(lst, element):
    lst = np.asarray(lst)
    idx = (np.abs(lst - element)).argmin()
    return lst[idx]


def check_dimensions_are_equal(tens1, tens2):
    if tens1.shape[0] != tens2.shape[0] or tens1.shape[1] != tens2.shape[1]:
        raise ValueError("Tensors should have the same size.")


def is_2d_tensor(tens):
    if len(tens.shape) != 2:
        raise RuntimeError("The tensor should be 2d.")


def to_tensor(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_file(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, pd.DataFrame):
        return torch.from_numpy(obj.values)
    raise TypeError("Please give either a nd.array or torch.Tensor or pd.DataFrame")
