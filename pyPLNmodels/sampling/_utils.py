import torch
from scipy.linalg import toeplitz

from pyPLNmodels._data_handler import _format_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _format_dict_of_array(dict_array):
    for array in dict_array.values():
        array = _format_data(array)
    return dict_array


def _get_exog(n_samples, nb_cov, add_const, seed=0):
    if nb_cov == 0:
        return None
    torch.manual_seed(seed)
    indices = torch.multinomial(torch.ones(nb_cov), n_samples, replacement=True)
    exog = torch.nn.functional.one_hot(indices, num_classes=nb_cov).float().to(DEVICE)
    if add_const is True:
        exog[:, -1] = 0.2 * torch.randn(n_samples).to(
            DEVICE
        )  # avoid rank error when adding const
    return exog


def _get_coef(nb_cov, dim, mean, add_const, seed=0):
    if add_const is True:
        nb_cov += 1
    if nb_cov == 0:
        return None
    torch.manual_seed(seed)
    return torch.randn(nb_cov, dim, device=DEVICE) + mean


def _get_covariance(dim, seed=0):
    torch.manual_seed(seed)
    parameter_toeplitz = 0.1 * torch.rand(1).to(DEVICE) + 0.8
    to_toeplitz = parameter_toeplitz ** (torch.arange(dim, device=DEVICE))
    return (
        torch.from_numpy(toeplitz(to_toeplitz.cpu().numpy())) + 0.5 * torch.eye(dim)
    ).to(DEVICE)


def _get_offsets(n_samples, dim, use_offsets):
    if use_offsets is False:
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
