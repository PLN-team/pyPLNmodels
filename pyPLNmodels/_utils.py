import math  # pylint:disable=[C0114]
import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg as TLA
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import transforms
from patsy import dmatrices

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class _PlotArgs:
    def __init__(self, window):
        self.window = window
        self.running_times = []
        self.criterions = [1] * window
        self._elbos_list = []

    @property
    def iteration_number(self):
        return len(self._elbos_list)

    def _show_loss(self, ax=None, name_doss=""):
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
            -np.array(self._elbos_list),
            label="Negative ELBO",
        )
        last_elbos = np.round(self._elbos_list[-1], 6)
        ax.set_title(f"Negative ELBO. Best ELBO ={last_elbos}")
        ax.set_yscale("log")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("ELBO")
        ax.legend()

    def _show_stopping_criteration(self, ax=None):
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


def _init_covariance(counts, covariates, coef):
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
    log_y_centered = log_y - torch.mean(log_y, axis=0)
    # MLE in a Gaussian setting
    n_samples = counts.shape[0]
    sigma_hat = 1 / (n_samples - 1) * (log_y_centered.T) @ log_y_centered
    return sigma_hat


def _init_components(counts, covariates, coef, rank):
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
    sigma_hat = _init_covariance(counts, covariates, coef).detach()
    components = _components_from_covariance(sigma_hat, rank)
    return components


def _init_latent_mean(
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


def _sigmoid(tens):
    """Compute the _sigmoid function of x element-wise."""
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
    if covariates is None:
        XB = 0
    else:
        XB = covariates @ coef
    gaussian = torch.mm(torch.randn(n_samples, rank, device=DEVICE), components.T) + XB
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


def _components_from_covariance(covariance, rank):
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


def _init_coef(counts, covariates, offsets):
    if covariates is None:
        return None
    poiss_reg = _PoissonReg()
    poiss_reg.fit(counts, covariates, offsets)
    return poiss_reg.beta


def _log_stirling(integer):
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
    components_posterior_mean = torch.matmul(
        components.unsqueeze(0), posterior_mean.unsqueeze(2)
    ).squeeze()
    if covariates is None:
        XB = 0
    else:
        XB = covariates @ coef
    log_lambda = offsets + components_posterior_mean + XB
    first_term = (
        -rank / 2 * math.log(2 * math.pi)
        - 1 / 2 * torch.norm(posterior_mean, dim=-1) ** 2
    )
    second_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * counts - _log_stirling(counts), axis=-1
    )
    return first_term + second_term


def _trunc_log(tens, eps=1e-16):
    integer = torch.min(torch.max(tens, torch.tensor([eps])), torch.tensor([1 - eps]))
    return torch.log(integer)


def _get_offsets_from_sum_of_counts(counts):
    sum_of_counts = torch.sum(counts, axis=1)
    return sum_of_counts.repeat((counts.shape[1], 1)).T


def _raise_wrong_dimension_error(
    str_first_array, str_second_array, dim_first_array, dim_second_array, dim_of_error
):
    msg = (
        f"The size of tensor {str_first_array} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_of_error}"
    )
    raise ValueError(msg)


def _check_two_dimensions_are_equal(
    str_first_array, str_second_array, dim_first_array, dim_second_array, dim_of_error
):
    if dim_first_array != dim_second_array:
        _raise_wrong_dimension_error(
            str_first_array,
            str_second_array,
            dim_first_array,
            dim_second_array,
            dim_of_error,
        )


def _init_S(counts, covariates, offsets, beta, C, M):
    n, rank = M.shape
    batch_matrix = torch.matmul(C.unsqueeze(2), C.unsqueeze(1)).unsqueeze(0)
    CW = torch.matmul(C.unsqueeze(0), M.unsqueeze(2)).squeeze()
    common = torch.exp(offsets + covariates @ beta + CW).unsqueeze(2).unsqueeze(3)
    prod = batch_matrix * common
    hess_posterior = torch.sum(prod, axis=1) + torch.eye(rank).to(DEVICE)
    inv_hess_posterior = -torch.inverse(hess_posterior)
    hess_posterior = torch.diagonal(inv_hess_posterior, dim1=-2, dim2=-1)
    return hess_posterior


def _format_data(data):
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


def _format_model_param(counts, covariates, offsets, offsets_formula, take_log_offsets):
    counts = _format_data(counts)
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


def _remove_useless_intercepts(covariates):
    covariates = _format_data(covariates)
    if covariates.shape[1] < 2:
        return covariates
    first_column = covariates[:, 0]
    second_column = covariates[:, 1]
    diff = first_column - second_column
    if torch.sum(torch.abs(diff - diff[0])) == 0:
        print("removing one")
        return covariates[:, 1:]
    return covariates


def _check_data_shape(counts, covariates, offsets):
    n_counts, p_counts = counts.shape
    n_offsets, p_offsets = offsets.shape
    _check_two_dimensions_are_equal("counts", "offsets", n_counts, n_offsets, 0)
    if covariates is not None:
        n_cov, _ = covariates.shape
        _check_two_dimensions_are_equal("counts", "covariates", n_counts, n_cov, 0)
    _check_two_dimensions_are_equal("counts", "offsets", p_counts, p_offsets, 1)


def _nice_string_of_dict(dictionnary):
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            return_string += f"{str(element):>12}"
        return_string += "\n"
    return return_string


def _plot_ellipse(mean_x, mean_y, cov, ax):
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


def _get_components_simulation(dim, rank):
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
    n_samples=100, dim=25, rank=5, nb_cov=1, return_true_param=False, seed=0
):
    components = _get_components_simulation(dim, rank)
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


def _closest(lst, element):
    lst = np.asarray(lst)
    idx = (np.abs(lst - element)).argmin()
    return lst[idx]


class _PoissonReg:
    """Poisson regressor class."""

    def __init__(self):
        """No particular initialization is needed."""
        pass

    def fit(self, Y, covariates, O, Niter_max=300, tol=0.001, lr=0.005, verbose=False):
        """Run a gradient ascent to maximize the log likelihood, using
        pytorch autodifferentiation. The log likelihood considered is
        the one from a poisson regression model. It is roughly the
        same as PLN without the latent layer Z.

        Args:
                        Y: torch.tensor. Counts with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            Niter_max: int, optional. The maximum number of iteration.
                Default is 300.
            tol: non negative float, optional. The tolerance criteria.
                Will stop if the norm of the gradient is less than
                or equal to this threshold. Default is 0.001.
            lr: positive float, optional. Learning rate for the gradient ascent.
                Default is 0.005.
            verbose: bool, optional. If True, will print some stats.

        Returns : None. Update the parameter beta. You can access it
                by calling self.beta.
        """
        # Initialization of beta of size (d,p)
        beta = torch.rand(
            (covariates.shape[1], Y.shape[1]), device=DEVICE, requires_grad=True
        )
        optimizer = torch.optim.Rprop([beta], lr=lr)
        i = 0
        grad_norm = 2 * tol  # Criterion
        while i < Niter_max and grad_norm > tol:
            loss = -_compute_poissreg_log_like(Y, O, covariates, beta)
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
                    print("Maxium number of iterations reached")
        self.beta = beta


def _compute_poissreg_log_like(Y, O, covariates, beta):
    """Compute the log likelihood of a Poisson regression."""
    # Matrix multiplication of X and beta.
    XB = torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    # Returns the formula of the log likelihood of a poisson regression model.
    return torch.sum(-torch.exp(O + XB) + torch.multiply(Y, O + XB))


def _to_tensor(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, pd.DataFrame):
        return torch.from_numpy(obj.values)
    raise TypeError("Please give either a nd.array or torch.Tensor or pd.DataFrame")


def _check_dimensions_are_equal(tens1, tens2):
    if tens1.shape[0] != tens2.shape[0] or tens1.shape[1] != tens2.shape[1]:
        raise ValueError("Tensors should have the same size.")


def load_model(path_of_directory):
    working_dict = os.getcwd()
    os.chdir(path_of_directory)
    all_files = os.listdir()
    data = {}
    for filename in all_files:
        if len(filename) > 4:
            if filename[-4:] == ".csv":
                parameter = filename[:-4]
                try:
                    data[parameter] = pd.read_csv(filename, header=None).values
                except pd.errors.EmptyDataError as err:
                    print(
                        f"Can't load {parameter} since empty. Standard initialization will be performed"
                    )
    os.chdir(working_dict)
    return data


def load_pln(path_of_directory):
    return load_model(path_of_directory)


def load_plnpca(path_of_directory, ranks=None):
    working_dict = os.getcwd()
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
        datas[rank] = load_model(f"_PLNPCA_rank_{rank}")
    os.chdir(working_dict)
    return datas


def _check_right_rank(data, rank):
    data_rank = data["latent_mean"].shape[1]
    if data_rank != rank:
        raise RuntimeError(
            f"Wrong rank during initialization."
            f" Got rank {rank} and data with rank {data_rank}."
        )


def _extract_data_from_formula(formula, data):
    dmatrix = dmatrices(formula, data=data)
    counts = dmatrix[0]
    covariates = dmatrix[1]
    if covariates.size == 0:
        covariates = None
    offsets = data.get("offsets", None)
    return counts, covariates, offsets


def _is_dict_of_dict(dictionnary):
    if isinstance(dictionnary[list(dictionnary.keys())[0]], dict):
        return True
    return False


def _get_dict_initalization(rank, dict_of_dict):
    if dict_of_dict is None:
        return None
    return dict_of_dict[rank]
