import torch
import numpy as np

from pyPLNmodels.base import BaseModel
from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels.elbos import profiled_elbo_pln


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Pln(BaseModel):
    """Simplest model, that is the original PLN model from
    Aitchison, J., and C. H. Ho. “The Multivariate Poisson-Log Normal Distribution.” Biometrika.
    """

    def _init_model_parameters(self):
        """The model parameters are profiled in the ELBO, no need to intialize them."""

    def _init_latent_parameters(self):
        self._latent_mean = torch.log(self._endog + (self._endog == 0)).to(DEVICE)
        self._latent_sqrt_variance = (
            1 / 2 * torch.ones((self.n_samples, self.dim)).to(DEVICE)
        )

    @property
    def _description(self):
        return "full covariance."

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_variance]

    def compute_elbo(self):
        return profiled_elbo_pln(
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_variance,
        )

    @property
    def dict_model_parameters(self):
        return self._default_dict_model_parameters

    @property
    def dict_latent_parameters(self):
        return self._default_dict_latent_parameters

    @property
    def _coef(self):
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _covariance(self):
        return _closed_formula_covariance(
            self._marginal_mean,
            self._latent_mean,
            self._latent_sqrt_variance,
            self.n_samples,
        )

    def _get_two_dim_covariances(self, components):
        components_var = np.expand_dims(
            self.latent_sqrt_variance**2, 1
        ) * np.expand_dims(components, 0)
        covariances = np.matmul(components_var, np.expand_dims(components.T, 0))
        return covariances
