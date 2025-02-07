from typing import Dict

import torch

from pyPLNmodels._data_handler import _add_constant_to_exog

from ._utils import _get_coef, _get_diag_covariance, _get_exog, _get_offsets


class PlnMixtureSampler:  # pylint: disable=too-many-instance-attributes
    """
    Initalize the data and parameters of the model.


    Examples
    --------
    >>> from pyPLNmodels import PlnMixtureSampler, PlnMixture
    >>> sampler = PlnMixtureSampler()
    >>> endog = sampler.sample()
    >>> mixture = PlnMixture(endog, exog = sampler.exog, add_const = False, n_cluster = sampler.n_cluster)#pylint:disable = line-too-long
    >>> mixture.fit()
    >>> estimated_covs = pln.covariances
    >>> true_covariances = sampler.covariances
    >>> estimated_latent_var = pln.latent_variables
    >>> true_latent_var = sampler.latent_variables
    """

    latent_variables: torch.Tensor
    clusters: torch.Tensor

    def __init__(
        self,
        *,
        n_cluster=3,
        nb_cov=1,
        dim=20,
        n_samples=100,
        add_const=True,
        add_offsets=False,
        seed=0,
    ):  # pylint: disable=too-many-arguments
        if nb_cov == 0 and add_const is False:
            raise ValueError("No mean in the model.")
        self.n_cluster = n_cluster
        self.dim = dim
        torch.manual_seed(seed)
        self.n_samples = n_samples
        cluster_probs = torch.rand(n_cluster) + dim / 2
        cluster_probs /= cluster_probs.sum()
        coefs = {}
        covariances = {}
        for i in range(self.n_cluster):
            coefs[i] = _get_coef(
                nb_cov=nb_cov,
                dim=dim,
                mean=i + 1,
                add_const=add_const,
                seed=(seed + 1) * i,
            )
            covariances[i] = _get_diag_covariance(dim, seed=(seed + 1) * i)
        self._exog_no_add = _get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const, seed=seed
        )
        self._offsets = _get_offsets(
            n_samples=n_samples, dim=dim, add_offsets=add_offsets, seed=seed
        )
        if add_const is True:
            self._exog = _add_constant_to_exog(self._exog_no_add, n_samples)
        else:
            self._exog = self._exog_no_add
        self._params = {}
        self._params["covariances"] = covariances
        self._params["coefs"] = coefs
        self._params["cluster_probs"] = cluster_probs

    @property
    def params(self) -> Dict[str, torch.Tensor]:
        """Method for the parameters of the model."""
        return {key: param.cpu() for key, param in self._params.items()}

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
    def covariances(self) -> torch.tensor:
        """Covariance matrix of each cluster."""
        covariances = self._params.get("covariances")
        return {key: value.cpu() for key, value in covariances.items()}

    @property
    def coefs(self) -> torch.tensor:
        """Coefficient matrix of each cluster. Returns a dictionnary"""
        coefs = self._params.get("coefs")
        return {key: value.cpu() for key, value in coefs.items()}

    @property
    def cluster_probs(self):
        """The cluster probabilites of the model. Returns a torch.Tensor."""
        return self._params.get("cluster_probs")

    # @property
    # def marginal_means(self):
    def sample(self, seed: int = 0):
        """
        Generate samples from the mixture model.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generation, by default 0.

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        torch.manual_seed(seed)
        gaussians = torch.randn(self.n_samples, self.dim)
        self.clusters = torch.multinomial(
            self.cluster_probs, self.n_samples, replacement=True
        )
        # gaussians += torch.matmul(self._exog, se)
        for cluster_number in range(self.n_cluster):
            indices = self.clusters == cluster_number
            gaussians[indices] *= self._params["covariances"][cluster_number]
            gaussians[indices] += (
                self._exog[indices] @ self._params["coefs"][cluster_number]
            )
        gaussians += self._offsets
        self.latent_variables = gaussians
        endog = torch.poisson(torch.exp(gaussians)).int()
        return endog.cpu()
