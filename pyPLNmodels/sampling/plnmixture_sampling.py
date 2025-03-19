import torch

from pyPLNmodels.utils._utils import _add_doc


from ._base_sampler import _BaseSampler
from ._utils import (
    _get_coef,
    _get_diag_covariance,
    _get_exog,
    _get_offsets,
    _get_mean,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnMixtureSampler(_BaseSampler):  # pylint: disable=too-many-instance-attributes
    """
    Initalize the data and parameters of the PlnMixture model.
    This is basically a Poisson model where the log intensity
    is given by a GMM.

    Examples
    --------
    >>> from pyPLNmodels import PlnMixtureSampler, PlnMixture
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> sampler = PlnMixtureSampler(nb_cov = 0, dim = 2, n_samples = 300)
    >>> endog = sampler.sample()
    >>> gmm = sampler.latent_variables
    >>> fig, axes = plt.subplots(3)
    >>> sns.scatterplot(x = gmm[:,0], y = gmm[:,1], hue = sampler.clusters, ax = axes[0])
    >>> sns.scatterplot(x = endog[:,0], y = endog[:,1], hue = sampler.clusters, ax = axes[1])
    >>> axes[0].set_title("Clusters in latent space")
    >>> axes[1].set_title("Clusters in integer space")
    >>> mixture = PlnMixture(endog, exog = sampler.exog, n_cluster = sampler.n_cluster)#pylint:disable = line-too-long
    >>> mixture.fit()
    >>> mixture.viz(ax = axes[2])
    >>> axes[2].set_title( "Inferred clusters and latent variables")
    >>> plt.show()


    See also
    --------
    :class:`pyPLNmodels.PlnMixture`
    """

    latent_variables: torch.Tensor
    clusters: torch.Tensor
    _clusters: torch.Tensor

    def __init__(
        self,
        n_samples=100,
        dim=20,
        *,
        nb_cov=1,
        add_const: bool = False,
        add_offsets=False,
        n_cluster=3,
        seed=0,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        if add_const is True:
            msg = "The `add_const` keyword is useless here. Adding "
            msg += "an intercept in the covariates results in non-identifiable coefficients."
            msg += "Set `add_const` to False."
            raise ValueError(msg)
        self.n_cluster = n_cluster
        torch.manual_seed(seed)
        self.n_samples = n_samples
        weights = torch.rand(n_cluster)
        weights /= weights.sum()
        cluster_bias = []
        covariances = []
        for i in range(self.n_cluster):
            cluster_bias.append(_get_mean(dim=dim, mean=2 * i + 1, seed=(seed + 1) * i))
            covariances.append(_get_diag_covariance(dim, seed=(seed + 1) * i))
        cluster_bias = torch.stack(cluster_bias, dim=0)
        covariances = torch.stack(covariances, dim=0)
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

    @property
    def covariances(self) -> torch.tensor:
        """Covariance matrix of each cluster."""
        return self._params.get("covariances")

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
        return self._params.get("cluster_bias")

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
        gaussians = torch.randn(self.n_samples, self.dim, device=DEVICE)
        self._clusters = torch.multinomial(
            self.weights, self.n_samples, replacement=True
        ).to(DEVICE)
        for cluster_number in range(self.n_cluster):
            indices = self._clusters == cluster_number
            gaussians[indices] *= torch.sqrt(
                self._params["covariances"][cluster_number]
            )
            gaussians[indices] += self._params["cluster_bias"][cluster_number]
        gaussians += self._marginal_mean
        return gaussians

    @property
    def clusters(self):
        """
        The clusters that LDA will be based on.
        """
        return self._clusters.tolist()

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return torch.matmul(self._exog, self._params["coef"])

    @property
    def marginal_mean(self):
        """
        Marginal mean of the latent variables, not knwowing the endog variables.
        """
        if self._exog is None:
            return 0
        return self._marginal_mean.cpu()
