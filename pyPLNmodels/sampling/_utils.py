import torch
import numpy as np
from scipy.linalg import toeplitz

from pyPLNmodels._data_handler import _format_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _format_dict_of_array(dict_array):
    for array in dict_array.values():
        array = _format_data(array)
    return dict_array


def _get_exog(n_samples, nb_cov):
    if nb_cov == 0:
        return None
    return (
        torch.from_numpy(
            np.random.multinomial(1, [1 / nb_cov] * nb_cov, size=n_samples)
        )
        .float()
        .to(DEVICE)
    )


def _get_coef(nb_cov, dim, mean):
    if nb_cov == 0:
        return None
    return torch.randn(nb_cov, dim, device=DEVICE) + mean


def _get_covariance(dim):
    parameter_toeplitz = 0.1 * torch.rand(1).to(DEVICE) + 0.8
    to_toeplitz = parameter_toeplitz ** (torch.arange(dim, device=DEVICE))
    return (
        torch.from_numpy(toeplitz(to_toeplitz.cpu().numpy())) + 0.5 * torch.eye(dim)
    ).to(DEVICE)


def _get_offsets(n_samples, dim, use_offsets):
    if use_offsets is False:
        return torch.zeros(n_samples, dim).to(DEVICE)
    return torch.rand(n_samples, dim).to(DEVICE)
