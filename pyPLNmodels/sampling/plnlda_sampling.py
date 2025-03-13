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
    >>> ntrain, ntest = 300, 200
    >>> nb_cov, n_cluster = 1,3
    >>> sampler = PlnLDASampler(
    >>> n_samples=ntrain + ntest, nb_cov=nb_cov, n_cluster=n_cluster, add_const=False)
    >>> endog = sampler.sample()
    >>> known_exog = sampler.known_exog
    >>> clusters = sampler.clusters
    >>> endog_train, endog_test = endog[:ntrain], endog[ntrain:]
    >>> known_exog_train, known_exog_test = known_exog[:ntrain], known_exog[ntrain:]
    >>> clusters_train, clusters_test = clusters[:ntrain],clusters[ntrain:]
    >>> lda = PlnLDA(endog_train,
    >>>    clusters = clusters_train, exog = known_exog_train, add_const = False).fit()
    >>> pred_train = lda.predict_clusters(endog_train, exog = known_exog_train)
    >>> mean_right_pred_train = torch.mean((torch.tensor(pred_train)==clusters_train).float())
    >>> print('Pourcentage of right predictions on train set:', mean_right_pred_train)
    >>> pred_test = lda.predict_clusters(endog_test, exog = known_exog_test)
    >>> mean_right_pred_test = torch.mean((torch.tensor(pred_test)==clusters_test).float())
    >>> print('Pourcentage of right predictions on test set:', mean_right_pred_test)


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
        n_cluster: int = 2,
        add_const: bool = False,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        seed: int = 0,
    ):  # pylint: disable=too-many-arguments
        self.n_cluster = n_cluster
        if add_const is True:
            raise ValueError(
                "`add_const` can not be set to `True` (for inversibility purposes)"
            )
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov + n_cluster,
            add_const=add_const,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            seed=seed,
        )
        self._params["coef_clusters"] = torch.clone(
            self._params["coef"][(-self.n_cluster) :]
        )
        if self._exog is None:
            self._params["coef"] = None
        else:
            self._params["coef"] = torch.clone(
                self._params["coef"][: (-self.n_cluster)]
            )

    @property
    def _marginal_mean(self):
        _marginal_mean_clusters = self._exog_clusters @ self._params["coef_clusters"]
        if self._known_exog is None:
            return _marginal_mean_clusters
        return _marginal_mean_clusters + self._known_exog @ self._params["coef"]

    def _get_exog(self, *, n_samples, nb_cov, will_add_const, seed):
        known_exog = _get_exog(
            n_samples=n_samples,
            nb_cov=nb_cov - self.n_cluster,
            will_add_const=True,
            seed=seed,
        )
        cluster_exog = _get_exog(
            n_samples=n_samples,
            nb_cov=self.n_cluster,
            will_add_const=False,
            seed=seed + 1,
        )
        if known_exog is None:
            return cluster_exog
        return torch.cat((known_exog, cluster_exog), dim=1)

    def _get_coef(
        self, *, nb_cov, dim, mean, add_const, seed
    ):  # pylint:disable = too-many-arguments
        nb_cov = nb_cov - self.n_cluster
        coef_known = super()._get_coef(
            nb_cov=nb_cov, dim=dim, mean=mean, add_const=add_const, seed=seed
        )
        coef_clusters = super()._get_coef(
            nb_cov=self.n_cluster, dim=dim, mean=0, add_const=False, seed=seed + 1
        )
        for i in range(self.n_cluster):
            coef_clusters[i] += i
        if nb_cov == 0:
            return coef_clusters
        return torch.cat((coef_known, coef_clusters), dim=0)

    @property
    def coef_clusters(self):
        """Regresson coefficients associated to the clusters."""
        return self._params["coef_clusters"].cpu()

    @property
    def _known_exog(self):
        if self._exog.shape[1] == self.n_cluster:
            return None
        return self._exog[:, : -(self.n_cluster)]

    @property
    def known_exog(self):
        """
        The exogenous varaibles that are always known in the model.
        """
        exog_device = self._known_exog
        if exog_device is None:
            return None
        return self._known_exog.cpu()

    @property
    def exog_clusters(self):
        """
        Clusters given in initialization.
        """
        return self._exog_clusters.cpu()

    @property
    def _exog_clusters(self):
        """
        Clusters given in initialization.
        """
        return self._exog[:, -(self.n_cluster) :]

    @property
    def clusters(self):
        """The cluster of each individual in the dataset."""
        return torch.argmax(self.exog_clusters, dim=1)
