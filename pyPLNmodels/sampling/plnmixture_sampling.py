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
    _get_mean,
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
    >>> mixture = PlnMixture(endog, exog = sampler.exog, n_clusters = sampler.n_clusters)#pylint:disable = line-too-long
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
        add_offsets=False,
        n_clusters=3,
        seed=0,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        self.n_clusters = n_clusters
        torch.manual_seed(seed)
        self.n_samples = n_samples
        weights = torch.rand(n_clusters)
        weights /= weights.sum()
        cluster_bias = {}
        covariances = {}
        for i in range(self.n_clusters):
            cluster_bias[i] = _get_mean(
                dim=dim,
                mean=i + 0.5,
                seed=(seed + 1) * i,
            )
            covariances[i] = _get_diag_covariance(dim, seed=(seed + 1) * i)
        exog_no_add = _get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=True, seed=seed
        )
        coef = _get_coef(nb_cov=nb_cov, mean=1, dim=dim, add_const=False, seed=seed)
        offsets = _get_offsets(
            n_samples=n_samples, dim=dim, add_offsets=add_offsets, seed=seed
        )
        params = {}
        params["covariances"] = covariances
        params["coef"] = coef
        params["weights"] = weights
        params["cluster_bias"] = cluster_bias
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            exog=exog_no_add,
            add_const=False,
            offsets=offsets,
            params=params,
        )

    def _format_parameters(self, params):
        covariances_params = _format_dict_of_array(params["covariances"])
        coefs_params = _format_data(params["coef"])
        weights_params = _format_data(params["weights"])
        cluster_bias = _format_dict_of_array(params["cluster_bias"])
        return {
            "covariances": covariances_params,
            "coef": coefs_params,
            "weights": weights_params,
            "cluster_bias": cluster_bias,
        }

    @property
    def covariances(self) -> torch.tensor:
        """Covariance matrix of each cluster."""
        covariances = self._params.get("covariances")
        return {key: value.cpu() for key, value in covariances.items()}

    @property
    def coef(self) -> torch.tensor:
        """Coefficient matrix that is common to each cluster. Returns a torch.Tensor."""
        return self._params.get("coef")

    @property
    def weights(self):
        """The cluster probabilites of the model. Returns a torch.Tensor."""
        return self._params.get("weights")

    @property
    def cluster_bias(self):
        """Mean vector depending on the cluster."""
        cluster_bias = self._params.get("cluster_bias")
        return {key: value.cpu() for key, value in cluster_bias.items()}

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
        torch.manual_seed(8)
        gaussians = torch.randn(self.n_samples, self.dim)
        self.clusters = torch.multinomial(
            self.weights, self.n_samples, replacement=True
        )
        for cluster_number in range(self.n_clusters):
            indices = self.clusters == cluster_number
            gaussians[indices] *= torch.sqrt(
                self._params["covariances"][cluster_number]
            )
            gaussians[indices] += self._params["cluster_bias"][cluster_number]
        gaussians += self._marginal_mean
        gaussians += self._offsets
        return gaussians

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return torch.matmul(self._exog, self._params["coef"])
