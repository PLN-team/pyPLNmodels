import torch
from scipy.linalg import toeplitz

from pyPLNmodels.utils._data_handler import _format_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _format_dict_of_array(dict_array):
    for array in dict_array.values():
        array = _format_data(array)
    return dict_array


def _get_exog(*, n_samples, nb_cov, will_add_const, seed):
    if nb_cov == 0:
        return None
    torch.manual_seed(seed)
    indices = torch.multinomial(torch.ones(nb_cov), n_samples, replacement=True)
    exog = torch.nn.functional.one_hot(indices, num_classes=nb_cov).float().to(DEVICE)
    if will_add_const is True or nb_cov == 1:
        exog[:, -1] = 0.2 * torch.randn(n_samples).to(
            DEVICE
        )  # avoid rank error when adding const
    return exog


def _get_coef(nb_cov, dim, mean, add_const, seed):
    if add_const is True:
        nb_cov += 1
    if nb_cov == 0:
        return None
    torch.manual_seed(seed)
    return torch.randn(nb_cov, dim, device=DEVICE) / 5 + mean


def _get_mean(dim, mean, seed):
    torch.manual_seed(seed)
    return torch.randn(dim, device=DEVICE) + mean


def _get_full_covariance(dim, seed):
    torch.manual_seed(seed)
    parameter_toeplitz = 0.1 * torch.rand(1).to(DEVICE) + 0.6
    to_toeplitz = parameter_toeplitz ** (torch.arange(dim, device=DEVICE))
    return (
        torch.from_numpy(toeplitz(to_toeplitz.cpu().numpy())) + 0.3 * torch.eye(dim)
    ).to(DEVICE)


def _get_covariance_ortho(ortho_components, seed):
    torch.manual_seed(seed)
    diag_cov = (torch.randn(ortho_components.shape[0]) ** 2 / 3 + 0.2).to(DEVICE)
    return ortho_components * diag_cov @ (ortho_components.T)


def _get_diag_covariance(dim, seed):
    torch.manual_seed(seed)
    return (
        torch.ones(dim, device=DEVICE) / 10 + torch.randn(dim, device=DEVICE) ** 2 / 5
    )


def _get_offsets(*, n_samples, dim, add_offsets, seed):
    torch.manual_seed(seed)
    if add_offsets is False:
        return torch.zeros(n_samples, dim).to(DEVICE)
    return torch.rand(n_samples, dim).to(DEVICE)


def _components_from_covariance(covariance, rank):
    """
    Compute the closest low-rank approximation of a covariance matrix
    i.e. reduces norm(covariance-components@components.T)

    Parameters
    ----------
    covariance: torch.tensor of size (dim,dim).
        Should be positive definite and symmetric.
    rank: int. The number of components.

    Returns:
        torch.tensor of size (dim,rank)
    """
    eigvalues, eigvectors = torch.linalg.eigh(covariance)
    components = eigvectors[:, -rank:] @ torch.diag(torch.sqrt(eigvalues[-rank:]))
    return components


def _random_zero_off_diagonal(matrix, proba):
    dim = matrix.shape[0]
    mask = torch.ones(dim, dim, dtype=torch.bool, device=DEVICE)
    mask.fill_diagonal_(0)

    # Generate a random upper triangular matrix
    random_matrix = torch.triu(torch.rand(dim, dim).to(DEVICE), diagonal=1)
    # Mirror the upper triangular part to the lower triangular part to make it symmetric
    random_matrix = random_matrix + random_matrix.T

    zero_mask = (random_matrix < proba) & mask

    matrix[zero_mask] = 0
    return matrix


def _get_sparse_precision(covariance, percentage_zeros):
    dim = covariance.shape[0]
    precision = torch.inverse(covariance)
    noise = (torch.rand(dim, dim) - 0.5) * 0.3
    noise = (noise + noise.T) / 2
    precision += 2 * torch.eye(dim, device=DEVICE)
    precision += noise.to(DEVICE)
    precision = _random_zero_off_diagonal(precision, percentage_zeros)
    return precision
