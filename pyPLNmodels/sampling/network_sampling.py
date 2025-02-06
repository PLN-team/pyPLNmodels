import torch

from pyPLNmodels._utils import _add_doc

from .pln_sampling import PlnSampler
from ._utils import (
    _random_zero_off_diagonal,
    _get_exog,
    _get_coef,
    _get_covariance,
    _get_offsets,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnNetworkSampler(PlnSampler):
    """Sampling a Pln model with precision matrix containing lots of zeros.

    Examples
    --------
    >>> from pyPLNmodels import PlnNetworkSampler, PlnNetwork
    >>> sampler = PlnNetworkSampler()
    >>> endog = sampler.sample()
    >>> net = PlnNetwork(endog, exog = sampler.exog, add_const = False)
    >>> net.fit()
    >>> estimated_precision = net.precision
    >>> true_precision = sampler.precision
    """

    @_add_doc(
        PlnSampler,
        params="""
            percentage_zeros : float, optional(keyword-only)
                The number of zeros in the precision matrix (i.e. inverse covariance).
                Defaults to 0.5.
                """,
        raises="""
                ValueError
                    If the percentage_zeros is not between 0 and 1.
                """,
    )
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        use_offsets: bool = False,
        marginal_mean_mean: int = 2,
        percentage_zeros: float = 0.3,
    ):  # pylint: disable=too-many-arguments
        if percentage_zeros > 1 or percentage_zeros < 0:
            msg = f"percentage_zeros should be a probability (0<=p<=1), got {percentage_zeros}."
            raise ValueError(msg)
        exog = _get_exog(n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const)
        offsets = _get_offsets(n_samples=n_samples, dim=dim, use_offsets=use_offsets)
        coef = _get_coef(
            nb_cov=nb_cov, dim=dim, mean=marginal_mean_mean, add_const=add_const
        )
        covariance = 3 * _get_covariance(dim)
        omega = torch.inverse(covariance)
        noise = (torch.rand(dim, dim) - 0.5) * 0.3
        noise = (noise + noise.T) / 2
        omega += 2 * torch.eye(dim, device=DEVICE)
        omega += noise.to(DEVICE)
        omega = _random_zero_off_diagonal(omega, percentage_zeros)
        covariance = torch.inverse(omega)
        super(PlnSampler, self).__init__(
            n_samples=n_samples,
            exog=exog,
            add_const=add_const,
            offsets=offsets,
            coef=coef,
            covariance=covariance,
        )

    @property
    def precision(self):
        """Precision matrix of the model, that is the inverse covariance matrix."""
        return torch.inverse(self.covariance)
