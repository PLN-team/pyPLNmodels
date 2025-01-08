import torch

from pyPLNmodels.base import BaseModel


class Pln(BaseModel):
    """Simplest model, that is the original PLN model from
    Aitchison, J., and C. H. Ho. “The Multivariate Poisson-Log Normal Distribution.” Biometrika.
    """

    def _init_model_parameters(self):
        """The model parameters are profiled in the ELBO, no need to intialize them."""

    def _init_latent_parameters(self):
        self._latent_mean = torch.log(self._endog + (self._endog == 0))
        self._latent_sqrt_variance = 1 / 2 * torch.ones((self.n_samples, self.dim))

    @property
    def _description(self):
        return "full covariance."

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_variance]

    def compute_elbo(self):
        pass
