import torch

from pyPLNmodels.utils._utils import _add_doc

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import (
    _get_exog,
    _get_coef,
    _get_full_covariance,
    _get_offsets,
    _get_sparse_precision,
)

THRESHOLD = 1e-5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnNetworkSampler(PlnSampler):
    """Sampling a Pln model with precision matrix containing lots of zeros.

    Examples
    --------
    >>> from pyPLNmodels import PlnNetworkSampler, PlnNetwork
    >>> sampler = PlnNetworkSampler()
    >>> endog = sampler.sample()
    >>> net = PlnNetwork(endog, exog = sampler.exog, add_const = False, penalty = 1)
    >>> net.fit()
    >>> estimated_precision = net.precision
    >>> true_precision = sampler.precision

    See also
    --------
    :class:`pyPLNmodels.PlnNetwork`
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
        example="""
        >>> from pyPLNmodels import PlnNetworkSampler
        >>> sampler = PlnNetworkSampler()
        >>> endog = sampler.sample()
        """,
    )
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        percentage_zeros: float = 0.3,
        seed: int = 0,
    ):  # pylint: disable=too-many-arguments
        if percentage_zeros > 1 or percentage_zeros < 0:
            msg = f"percentage_zeros should be a probability (0<=p<=1), got {percentage_zeros}."
            raise ValueError(msg)
        exog = _get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const, seed=seed
        )
        offsets = _get_offsets(
            n_samples=n_samples, dim=dim, add_offsets=add_offsets, seed=seed
        )
        coef = _get_coef(
            nb_cov=nb_cov,
            dim=dim,
            mean=marginal_mean_mean,
            add_const=add_const,
            seed=seed,
        )
        covariance = 3 * _get_full_covariance(dim, seed=seed)
        precision = _get_sparse_precision(covariance, percentage_zeros)
        covariance = torch.inverse(precision)
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

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnNetworkSampler
        >>> sampler = PlnNetworkSampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)

    @property
    def nb_zeros_precision(self):
        """Number of zeros in the precision matrix without (on the lower diagonal)."""
        return torch.sum((torch.abs(self.precision) < THRESHOLD).float()) / 2
