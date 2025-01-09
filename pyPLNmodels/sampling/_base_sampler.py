from abc import ABC, abstractmethod
from typing import Dict

import torch

from pyPLNmodels._data_handler import _format_data

from ._utils import _format_dict_of_array


class _BaseSampler(ABC):
    """An abstract class used to simulate data (endogenous variables) from a model."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        n_samples: int,
        dim: int,
        exog: torch.Tensor,
        offsets: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ):
        """
        Instantiate the model with the data given.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        dim : int
            The number of dimension (i.e. number of features).
        exog : torch.Tensor
            Covariates with size (n, d).
        offsets : torch.Tensor
            Offsets with size (n, p).
        params : dict[str, torch.Tensor]
            Model parameters. Each item should a torch.Tensor.
        """
        self._n_samples: int = n_samples
        self._dim: int = dim
        self._exog: torch.Tensor = _format_data(exog)
        self._offsets: torch.Tensor = _format_data(offsets)
        self._params: Dict[str, torch.Tensor] = _format_dict_of_array(params)

    def sample(self, seed: int = 0) -> torch.Tensor:
        """
        Generate samples from the model.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generation, by default 0.

        Returns
        -------
        np.ndarray
            Generated samples.
        """
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        gaussians = self._get_gaussians()
        endog = torch.poisson(torch.exp(gaussians))
        torch.random.set_rng_state(prev_state)
        return endog

    @abstractmethod
    def _get_gaussians(self) -> torch.Tensor:
        """Method to generate the Gaussian samples."""

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._n_samples

    @property
    def dim(self) -> int:
        """Latent dimension."""
        return self._dim

    @property
    def params(self) -> Dict[str, torch.Tensor]:
        """Method for the parameters of the model."""
        return self._params.cpu()

    @property
    def exog(self) -> torch.Tensor:
        """Covariates."""
        return self._exog.cpu()

    @property
    def offsets(self) -> torch.Tensor:
        """Offsets."""
        return self._offsets.cpu()

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance matrix."""
        return self._params["covariance"].cpu()

    @property
    def coef(self) -> torch.Tensor:
        """Coefficient matrix."""
        return self._params["coef"].cpu()
