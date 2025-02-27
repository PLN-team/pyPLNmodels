import torch

from .pln_sampling import PlnSampler
from ._utils import _get_exog


class PlnLDASampler(PlnSampler):
    """
    Supervised Pln model Sampler. The classification is based on Linear Discriminant Analysis (LDA).
    No classification is made here, only sampling.
    See J. Chiquet, M. Mariadassou, S. Robin: "The Poisson-Lognormal Model as a Versatile Framework
    for the Joint Analysis of Species Abundances" for more information.

    Examples
    --------
    >>> import torch
    >>> from pyPLNmodels import PlnLDA, PlnLDASampler
    >>> ntrain, ntest = 1000, 200
    >>> nb_cov, n_clusters = 1,3
    >>> sampler = PlnLDASampler(
    >>> n_samples=ntrain + ntest, nb_cov=nb_cov, n_clusters=n_clusters, add_const=False, dim=300)
    >>> endog = sampler.sample()
    >>> known_exog = sampler.known_exog
    >>> clusters = sampler.clusters
    >>> endog_train, endog_test = endog[:ntrain], endog[ntrain:]
    >>> known_exog_train, known_exog_test = known_exog[:ntrain], known_exog[ntrain:]
    >>> clusters_train, clusters_test = clusters[:ntrain],clusters[ntrain:]
    >>> lda = PlnLDA(endog_train,
    >>>    clusters = clusters_train, exog = known_exog_train, add_const = False).fit()
    >>> pred = lda.predict_clusters(endog_test, exog = known_exog_test)
    >>> true_cluster = torch.argmax(clusters_test, dim = 1)
    >>> mean_right_pred = torch.mean((torch.tensor(pred)==true_cluster).float())
    >>> print('Pourcentage of right predictions:', mean_right_pred)

    See also
    --------
    :class:`pyPLNmodels.PlnLDA`
    :class:`pyPLNmodels.Pln`
    :class:`pyPLNmodels.PlnMixture`
    """

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        n_clusters: int = 2,
        add_const: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        seed: int = 0,
    ):  # pylint: disable=too-many-arguments
        self.n_clusters = n_clusters
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov + n_clusters,
            add_const=add_const,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            seed=seed,
        )

    def _get_exog(self, *, n_samples, nb_cov, will_add_const, seed):
        known_exog = _get_exog(
            n_samples=n_samples,
            nb_cov=nb_cov - self.n_clusters,
            will_add_const=will_add_const,
            seed=seed,
        )
        cluster_exog = _get_exog(
            n_samples=n_samples,
            nb_cov=self.n_clusters,
            will_add_const=False,
            seed=seed + 1,
        )
        if known_exog is None:
            return cluster_exog
        return torch.cat((known_exog, cluster_exog), dim=1)

    @property
    def known_exog(self):
        """
        The exogenous varaibles that are always known in the model.
        """
        return self._exog[:, : -(self.n_clusters)]

    @property
    def clusters(self):
        """
        Clusters given in initialization.
        """
        return (self._exog[:, -(self.n_clusters) :]).cpu()
