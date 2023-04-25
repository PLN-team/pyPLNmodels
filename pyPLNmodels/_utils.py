import math  # pylint:disable=[C0114]
from scipy.linalg import toeplitz

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg as TLA
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


torch.set_default_dtype(torch.float64)

# offsets is not doing anything in the initialization of Sigma. should be fixed.

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


def init_sigma(counts, covariates, offsets, beta):
    """Initialization for Sigma for the PLN model. Take the log of counts
    (careful when counts=0), remove the covariates effects X@beta and
    then do as a MLE for Gaussians samples.
    Args :
            counts: torch.tensor. Samples with size (n,p)
            0: torch.tensor. Offset, size (n,p)
            covariates: torch.tensor. Covariates, size (n,d)
            beta: torch.tensor of size (d,p)
    Returns : torch.tensor of size (p,p).
    """
    # Take the log of counts, and be careful when counts = 0. If counts = 0,
    # then we set the log(counts) as 0.
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    # we remove the mean so that we see only the covariances
    log_y_centered = (
        log_y - torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    )
    # MLE in a Gaussian setting
    n = counts.shape[0]
    Sigma_hat = 1 / (n - 1) * (log_y_centered.T) @ log_y_centered
    return Sigma_hat


def init_c(counts, covariates, offsets, beta, rank):
    """Inititalization for C for the PLN model. Get a first
    guess for Sigma that is easier to estimate and then takes
    the rank largest eigenvectors to get C.
    Args :
        counts: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covarites: torch.tensor. Covariates, size (n,d)
        beta: torch.tensor of size (d,p)
        rank: int. The dimension of the latent space, i.e. the reducted dimension.
    Returns :
        torch.tensor of size (p,rank). The initialization of C.
    """
    Sigma_hat = init_sigma(counts, covariates, offsets, beta).detach()
    C = C_from_Sigma(Sigma_hat, rank)
    return C


def init_M(counts, covariates, offsets, beta, C, N_iter_max=500, lr=0.01, eps=7e-3):
    """Initialization for the variational parameter M. Basically,
    the mode of the log_posterior is computed.

    Args:
        counts: torch.tensor. Samples with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        beta: torch.tensor of size (d,p)
        N_iter_max: int. The maximum number of iteration in
            the gradient ascent.
        lr: positive float. The learning rate of the optimizer.
        eps: positive float, optional. The tolerance. The algorithm will stop
            if the maximum of |W_t-W_{t-1}| is lower than eps, where W_t
            is the t-th iteration of the algorithm.This parameter
            changes a lot the resulting time of the algorithm. Default is 9e-3.
    """
    W = torch.randn(counts.shape[0], C.shape[1], device=DEVICE)
    W.requires_grad_(True)
    optimizer = torch.optim.Rprop([W], lr=lr)
    crit = 2 * eps
    old_W = torch.clone(W)
    keep_condition = True
    i = 0
    while i < N_iter_max and keep_condition:
        batch_loss = log_PW_given_Y(counts, covariates, offsets, W, C, beta)
        loss = -torch.mean(batch_loss)
        loss.backward()
        optimizer.step()
        crit = torch.max(torch.abs(W - old_W))
        optimizer.zero_grad()
        if crit < eps and i > 2:
            keep_condition = False
        old_W = torch.clone(W)
        i += 1
    return W


def sigmoid(tens):
    """Compute the sigmoid function of x element-wise."""
    return 1 / (1 + torch.exp(-tens))


def sample_PLN(C, beta, covariates, offsets, B_zero=None):
    """Sample Poisson log Normal variables. If B_zero is not None, the model will
    be zero inflated.

    Args:
        C: torch.tensor of size (p,rank). The matrix C of the PLN model
        beta: torch.tensor of size (d,p). Regression parameter.
        0: torch.tensor of size (n,p). Offsets.
        covariates : torch.tensor of size (n,d). Covariates.
        B_zero: torch.tensor of size (d,p), optional. If B_zero is not None,
             the ZIPLN model is chosen, so that it will add a
             Bernouilli layer. Default is None.
    Returns :
        counts: torch.tensor of size (n,p), the count variables.
        Z: torch.tensor of size (n,p), the gaussian latent variables.
        ksi: torch.tensor of size (n,p), the bernoulli latent variables
        (full of zeros if B_zero is None).
    """

    n = offsets.shape[0]
    rank = C.shape[1]
    Z = torch.mm(torch.randn(n, rank, device=DEVICE), C.T) + covariates @ beta
    parameter = torch.exp(offsets + Z)
    if B_zero is not None:
        print("ZIPLN is sampled")
        ZI_cov = covariates @ B_zero
        ksi = torch.bernoulli(1 / (1 + torch.exp(-ZI_cov)))
    else:
        ksi = 0
    counts = (1 - ksi) * torch.poisson(parameter)
    return counts, Z, ksi


def logit(tens):
    """logit function. If x is too close from 1, we set the result to 0.
    performs logit element wise."""
    return torch.nan_to_num(torch.log(x / (1 - tens)), nan=0, neginf=0, posinf=0)


def build_block_Sigma(p, block_size):
    """Build a matrix per block of size (p,p). There will be p//block_size+1
    blocks of size block_size. The first p//block_size ones will be the same
    size. The last one will have a smaller size (size (0,0)
    if p%block_size = 0).
    Args:
        p: int.
        block_size: int. Should be lower than p.
    Returns: a torch.tensor of size (p,p) and symmetric.
    """
    k = p // block_size  # number of matrices of size p//block_size.
    alea = np.random.randn(k + 1) ** 2 + 1
    Sigma = np.zeros((p, p))
    last_block_size = p - k * block_size
    for i in range(k):
        Sigma[
            i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size
        ] = alea[i] * toeplitz(0.7 ** np.arange(block_size))
    # Last block matrix.
    if last_block_size > 0:
        Sigma[-last_block_size:, -last_block_size:] = alea[k] * toeplitz(
            0.7 ** np.arange(last_block_size)
        )
    return Sigma


def C_from_Sigma(Sigma, rank):
    """Get the best matrix of size (p,rank) when Sigma is of
    size (p,p). i.e. reduces norm(Sigma-C@C.T)
    Args :
        Sigma: torch.tensor of size (p,p). Should be positive definite and
            symmetric.
        rank: int. The number of columns wanted for C

    Returns:
        C_reduct: torch.tensor of size (p,rank) containing the rank eigenvectors with
        largest eigenvalues.
    """
    w, v = TLA.eigh(Sigma)
    C_reduct = v[:, -rank:] @ torch.diag(torch.sqrt(w[-rank:]))
    return C_reduct


def init_beta(counts, covariates, offsets):
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    log_y = log_y.to(DEVICE)
    return torch.matmul(
        torch.inverse(torch.matmul(covariates.T, covariates)),
        torch.matmul(covariates.T, log_y),
    )


def log_stirling(n):
    """Compute log(n!) even for n large. We use the Stirling formula to avoid
    numerical infinite values of n!.
    Args:
         n: torch.tensor of any size.
    Returns:
        An approximation of log(n_!) element-wise.
    """
    n_ = n + (n == 0)  # Replace the 0 with 1. It doesn't change anything since 0! = 1!
    return torch.log(torch.sqrt(2 * np.pi * n_)) + n_ * torch.log(n_ / math.exp(1))


def log_PW_given_Y(counts_b, covariates_b, offsets_b, W, C, beta):
    """Compute the log posterior of the PLN model. Compute it either
    for W of size (N_samples, N_batch,rank) or (batch_size, rank). Need to have
    both cases since it is done for both cases after. Please the mathematical
    description of the package for the formula.
    Args :
        counts_b : torch.tensor of size (batch_size, p)
        covariates_b : torch.tensor of size (batch_size, d) or (d)
    Returns: torch.tensor of size (N_samples, batch_size) or (batch_size).
    """
    length = len(W.shape)
    rank = W.shape[-1]
    if length == 2:
        CW = torch.matmul(C.unsqueeze(0), W.unsqueeze(2)).squeeze()
    elif length == 3:
        CW = torch.matmul(C.unsqueeze(0).unsqueeze(1), W.unsqueeze(3)).squeeze()

    A_b = offsets_b + CW + covariates_b @ beta
    first_term = -rank / 2 * math.log(2 * math.pi) - 1 / 2 * torch.norm(W, dim=-1) ** 2
    second_term = torch.sum(
        -torch.exp(A_b) + A_b * counts_b - log_stirling(counts_b), axis=-1
    )
    return first_term + second_term


def trunc_log(tens, eps=1e-16):
    y = torch.min(torch.max(tens, torch.tensor([eps])), torch.tensor([1 - eps]))
    return torch.log(y)


def get_offsets_from_sum_of_counts(counts):
    sum_of_counts = torch.sum(counts, axis=1)
    return sum_of_counts.repeat((counts.shape[1], 1)).T


def raise_wrong_dimension_error(
    str_first_array, str_second_array, dim_first_array, dim_second_array, dim_of_error
):
    msg = (
        f"The size of tensor {str_first_array} ({dim_first_array}) must match"
        f"the size of tensor {str_second_array} ({dim_second_array}) at"
        f"non-singleton dimension {dim_of_error}"
    )
    raise ValueError(msg)


def check_dimensions_are_equal(
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


def init_S(counts, covariates, offsets, beta, C, M):
    n, rank = M.shape
    batch_matrix = torch.matmul(C.unsqueeze(2), C.unsqueeze(1)).unsqueeze(0)
    CW = torch.matmul(C.unsqueeze(0), M.unsqueeze(2)).squeeze()
    common = torch.exp(offsets + covariates @ beta + CW).unsqueeze(2).unsqueeze(3)
    prod = batch_matrix * common
    hess_posterior = torch.sum(prod, axis=1) + torch.eye(rank).to(DEVICE)
    inv_hess_posterior = -torch.inverse(hess_posterior)
    hess_posterior = torch.diagonal(inv_hess_posterior, dim1=-2, dim2=-1)
    return hess_posterior


def format_data(data):
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).double().to(DEVICE)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).double().to(DEVICE)
    if isinstance(data, torch.Tensor):
        return data
    raise AttributeError(
        "Please insert either a numpy array, pandas.DataFrame or torch.tensor"
    )


def check_parameters_shape(counts, covariates, offsets):
    n_counts, p_counts = counts.shape
    n_offsets, p_offsets = offsets.shape
    n_cov, _ = covariates.shape
    check_dimensions_are_equal("counts", "offsets", n_counts, n_offsets, 0)
    check_dimensions_are_equal("counts", "covariates", n_counts, n_cov, 0)
    check_dimensions_are_equal("counts", "offsets", p_counts, p_offsets, 1)


def extract_data(dictionnary, parameter_in_string):
    try:
        return dictionnary[parameter_in_string]
    except KeyError:
        return None


def extract_cov_offsets_offsetsformula(dictionnary):
    covariates = extract_data(dictionnary, "covariates")
    offsets = extract_data(dictionnary, "offsets")
    offsets_formula = extract_data(dictionnary, "offsets_formula")
    return covariates, offsets, offsets_formula


def nice_string_of_dict(dictionnary):
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            return_string += f"{str(element):>10}"
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
        alpha=0.1,
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
