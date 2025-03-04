from abc import ABC, abstractmethod
from typing import Dict

import torch

from pyPLNmodels.utils._data_handler import _format_data, _add_constant_to_exog

from ._utils import _format_dict_of_array


class _BaseSampler(ABC):  # pylint: disable=too-many-instance-attributes
    """An abstract class used to simulate data (endogenous variables) from a model."""

    latent_variables: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        n_samples: int,
        dim: int,
        exog: torch.Tensor,
        add_const: bool,
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
        self.n_samples = n_samples
        self.dim = dim
        self._exog_no_add: torch.Tensor = _format_data(exog)
        if add_const is True:
            self._exog = _add_constant_to_exog(self._exog_no_add, n_samples)
        else:
            self._exog = self._exog_no_add
        self.add_const: bool = add_const
        self._offsets: torch.Tensor = _format_data(offsets)
        self._params = self._format_parameters(params)

    def _format_parameters(self, params):
        return _format_dict_of_array(params)

    def sample(self, seed: int = 0) -> torch.Tensor:
        """
        Generate samples from the model.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generation, by default 0.

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        prev_state = torch.random.get_rng_state()
        gaussians = self._get_gaussians(seed=seed)
        endog = torch.poisson(torch.exp(self._offsets + gaussians))
        self.latent_variables = gaussians.cpu()
        torch.random.set_rng_state(prev_state)
        return endog.cpu()

    @abstractmethod
    def _get_gaussians(self, seed: int) -> torch.Tensor:
        """Method to generate the Gaussian samples."""

    @property
    def params(self) -> Dict[str, torch.Tensor]:
        """Method for the parameters of the model."""
        return {
            key: param.cpu() if param is not None else None
            for key, param in self._params.items()
        }

    @property
    def exog(self) -> torch.Tensor:
        """Exogenous variables (i.e. covariates)."""
        if self._exog is None:
            return None
        return self._exog.cpu()

    @property
    def exog_no_add(self) -> torch.Tensor:
        """Exogenous variables (i.e. covariates)."""
        if self._exog_no_add is None:
            return None
        return self._exog_no_add.cpu()

    @property
    def offsets(self) -> torch.Tensor:
        """Offsets."""
        return self._offsets.cpu()

    @property
    def dict_model_true_parameters(self):
        """Alias for the parameters."""
        return self.params
