import torch

from pyPLNmodels._data_handler import _format_data
from pyPLNmodels._utils import _add_doc

from ._base_sampler import _BaseSampler
from ._utils import (
    _get_coef,
    _get_diag_covariance,
    _get_exog,
    _get_offsets,
    _format_dict_of_array,
)


class PlnMixtureSampler(_BaseSampler):  # pylint: disable=too-many-instance-attributes
    """
    Initalize the data and parameters of the PLN Mixture model.
    This is basically a Poisson model where the log intensity
    is given by a GMM.

    Examples
    --------
    >>> from pyPLNmodels import PlnMixtureSampler, PlnMixture
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> sampler = PlnMixtureSampler(nb_cov = 0, dim = 2)
    >>> endog = sampler.sample()
    >>> gmm = sampler.latent_variables
    >>> fig, axes = plt.subplots(2)
    >>> sns.scatterplot(x = gmm[:,0], y = gmm[:,1], hue = sampler.clusters.numpy(), ax = axes[0])
    >>> sns.scatterplot(x = endog[:,0], y = endog[:,1], hue = sampler.clusters.numpy(), ax = axes[1])
    >>> mixture = PlnMixture(endog, exog = sampler.exog, add_const = False, n_cluster = sampler.n_cluster)#pylint:disable = line-too-long
    >>> mixture.fit()
    >>> mixture.viz()
    """

    latent_variables: torch.Tensor
    clusters: torch.Tensor

    def __init__(
        self,
        n_samples=100,
        dim=20,
        *,
        nb_cov=1,
        add_const=True,
        add_offsets=False,
        n_cluster=3,
        seed=0,
    ):  # pylint: disable=too-many-arguments
        if nb_cov == 0 and add_const is False:
            raise ValueError("No mean in the model.")
        self.n_cluster = n_cluster
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
        exog_no_add = _get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const, seed=seed
        )
        offsets = _get_offsets(
            n_samples=n_samples, dim=dim, add_offsets=add_offsets, seed=seed
        )
        params = {}
        params["covariances"] = covariances
        params["coefs"] = coefs
        params["cluster_probs"] = cluster_probs
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            exog=exog_no_add,
            add_const=add_const,
            offsets=offsets,
            params=params,
        )

    def _format_parameters(self, params):
        covariances_params = _format_dict_of_array(params["covariances"])
        coefs_params = _format_dict_of_array(params["coefs"])
        cluster_probs_params = _format_data(params["cluster_probs"])
        return {
            "covariances": covariances_params,
            "coefs": coefs_params,
            "cluster_probs": cluster_probs_params,
        }

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

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnMixtureSampler
        >>> sampler = PlnMixtureSampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0):
        return super().sample(seed=seed)

    def _get_gaussians(self, seed):
        torch.manual_seed(seed)
        gaussians = torch.randn(self.n_samples, self.dim)
        self.clusters = torch.multinomial(
            self.cluster_probs, self.n_samples, replacement=True
        )
        for cluster_number in range(self.n_cluster):
            indices = self.clusters == cluster_number
            gaussians[indices] *= self._params["covariances"][cluster_number]
            gaussians[indices] += (
                self._exog[indices] @ self._params["coefs"][cluster_number]
            )
        gaussians += self._offsets
        return gaussians
