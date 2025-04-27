from typing import Optional, Union
import warnings
import torch
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import silhouette_score

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.plndiag import PlnDiag
from pyPLNmodels.utils._data_handler import (
    _extract_data_from_formula,
    _check_dimensions_for_prediction,
    _check_full_rank_exog_and_ones,
    _check_int,
)
from pyPLNmodels.utils._utils import (
    _add_doc,
    _get_two_dim_latent_variances,
    _calculate_wcss,
)
from pyPLNmodels.calculations._initialization import _init_gmm
from pyPLNmodels.calculations.entropies import (
    entropy_gaussian_mixture,
    entropy_clusters,
)
from pyPLNmodels.calculations.elbos import per_sample_elbo_pln_mixture_diag
from pyPLNmodels.utils._viz import MixtureModelViz, _viz_variables


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnMixture(
    BaseModel
):  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """
    Pln mixture models, that is a gaussian mixture model with Poisson layer on top of it.
    The effect of covariates is shared with clusters. Note that stability
    may significantly decrease with the number of covariates.

    Examples
    --------
    >>> import seaborn as sns
    >>> from pyPLNmodels import PlnMixture, load_scrna, plot_confusion_matrix
    >>> data = load_scrna()
    >>> mixture = PlnMixture(data["endog"],n_cluster = 3)
    >>> mixture.fit()
    >>> print(mixture)
    >>> plot_confusion_matrix(mixture.clusters, data["labels"])


    See also
    --------
    :func:`pyPLNmodels.PlnDiag`
    :func:`pyPLNmodels.Pln`
    :func:`pyPLNmodels.PlnMixture.__init__`
    :func:`pyPLNmodels.PlnMixture.from_formula`
    :func:`pyPLNmodels.PlnMixture.predict
    """

    _weights: torch.Tensor
    _cluster_bias: torch.Tensor
    _latent_prob: torch.Tensor
    _latent_means: torch.Tensor
    _latent_sqrt_variances: torch.Tensor
    _sqrt_covariances: torch.Tensor
    _per_sample_per_cluster_elbo: torch.Tensor

    _ModelViz = MixtureModelViz

    @_add_doc(
        BaseModel,
        params="""
            n_cluster : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture(data["endog"],n_cluster = 3)
            >>> mixture.fit()
            >>> print(mixture)
        """,
        returns="""
            PlnMixture
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture.from_formula`
        """,
        notes="""
        The `add_const` keyword is useless here and ignored. Adding an intercept in the covariates
        results in non-identifiable coefficients for the mixture model.
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        n_cluster: int,
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        add_const: bool = False,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable=too-many-arguments
        _check_int(n_cluster)
        if add_const is True:
            msg = "You shall not use the add_const keyword for PlnMixture. Adding "
            msg += "an intercept in the covariates results in non-identifiable coefficients. "
            msg += "This is ignored."
            warnings.warn(msg)
        self._n_cluster = n_cluster
        self._compute_offsets_method = (
            compute_offsets_method  # store it for prediction after.
        )
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )
        if self._exog is not None:
            _exog_and_ones = torch.cat(
                (self._exog, torch.ones(self.n_samples).unsqueeze(1).to(DEVICE)), dim=1
            )
            _check_full_rank_exog_and_ones(_exog_and_ones)

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
            n_cluster : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture.from_formula("endog ~ 0",data, n_cluster = 3)
            >>> mixture.fit()
            >>> print(mixture)
        """,
        returns="""
            PlnMixture
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture`
        """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        n_cluster: int,
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable = arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            n_cluster=n_cluster,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_cluster = 3)
        >>> mixture.fit()
        >>> print(mixture)

        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_cluster = 3)
        >>> mixture.fit(maxiter=500, verbose=True)
        >>> print(mixture)
        """,
        returns="""
        PlnMixture object
        """,
    )
    def fit(
        self,
        *,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    def _covariance(self):
        raise NotImplementedError("No single covariance. Call ._covariances.")

    @property
    def _covariances(self):
        return self._sqrt_covariances**2

    @property
    def covariances(self):
        """
        Tensor of covariances of shape (n_cluster, dim).
        Each vector corresponds to the (diagonal) covariance of each cluster.
        """
        return self._covariances.detach().cpu()

    @property
    def weights(self):
        """
        Probability of a sample to belong to each cluster.
        Tensor of size (n_cluster).
        """
        return self._weights.cpu()

    def viz(
        self,
        *,
        ax=None,
        colors=None,
        show_cov: bool = False,
        remove_exog_effect: bool = False,
    ):
        """
        Visualize the latent variables. One can remove the effect of exogenous variables
        with the `remove_exog_effect` boolean variable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.
        colors : list, optional
            The labels to color the samples, of size `n_samples`.
            If None, will take the inferred clusters.
        show_cov : bool, optional
            Whether to show covariances, by default False.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_cluster = 3)
        >>> mixture.fit()
        >>> mixture.viz()
        >>> mixture.viz(colors=data["labels"])
        >>> mixture.viz(show_cov=True)
        >>> mixture.viz(remove_exog_effect=True, colors=data["labels"])

        See also
        --------
        :func:`pyPLNmodels.PlnMixture.biplot`
        :func:`pyPLNmodels.PlnMixture.predict_clusters`
        :func:`pyPLNmodels.PlnMixture.pca_pairplot`
        """
        clusters = self.clusters
        if colors is None:
            colors = clusters
        if show_cov is True:
            variables, covariances_per_cluster = (
                self._pca_projected_latent_variables_with_covariances(
                    remove_exog_effect=remove_exog_effect
                )
            )
            covariances = torch.zeros(self.n_samples, 2, 2)
            for k in range(self.n_cluster):
                indices = clusters == k
                covariances[indices] = covariances_per_cluster[k][indices]
        else:
            variables = self.projected_latent_variables(
                remove_exog_effect=remove_exog_effect
            )
            covariances = None
        _viz_variables(variables, ax=ax, colors=colors, covariances=covariances)

    @property
    def covariance(self):
        """The GMM does not have a single covariance. It has multiple covariances, one per cluster.
        You may call .covariances."""
        raise NotImplementedError(
            "The GMM does not have a single covariance. It has multiple covariances, one"
            " per cluster. You may call .covariances."
        )

    def _init_parameters(self):
        print("Intialization ...")
        pln = PlnDiag(
            endog=self._endog, exog=self._exog, offsets=self._offsets, add_const=True
        )
        pln.fit(maxiter=150)
        positions_with_mean = pln.latent_positions + (
            pln.exog[:, 0].unsqueeze(1) @ pln.coef[0].unsqueeze(0)
        )
        if pln.nb_cov > 1:
            _coef = pln.coef[:-1].to(DEVICE)  # Should not retrieve the mean
        else:
            _coef = None

        _cluster_bias, _sqrt_covariances, _weights, _latent_prob = _init_gmm(
            positions_with_mean, self.n_cluster
        )
        if not (
            hasattr(self, "_cluster_bias")
            or hasattr(self, "_sqrt_covariances")
            or hasattr(self, "_weights")
            or hasattr(self, "_latent_prob")
            or hasattr(self, "_coef")
        ):
            self._cluster_bias = _cluster_bias
            self._sqrt_covariances = _sqrt_covariances
            self._weights = _weights
            self._coef = _coef

        self._latent_prob = _latent_prob
        self._latent_means = self._cluster_bias.unsqueeze(1)
        if self._exog is not None:
            self._latent_means = self._latent_means + self._marginal_mean.unsqueeze(0)
        else:
            self._latent_means = self._latent_means.repeat_interleave(
                self.n_samples, dim=1
            )

        self._latent_sqrt_variances = (
            torch.randn(self.n_cluster, self.n_samples, self.dim).to(DEVICE) / 100
        )
        print("Finished!")

    def _init_latent_parameters(self):
        """Everything is done in the _init_parameters method."""

    def _init_model_parameters(self):
        """Everything is done in the _init_parameters method."""

    @property
    def _description(self):
        return f"diagonal covariances and {self.n_cluster} clusters."

    @property
    def _dict_for_printing(self):
        orig_dict = super()._dict_for_printing
        orig_dict["Silhouette"] = self.silhouette
        return orig_dict

    @property
    def _endog_predictions(self):
        exp_term = torch.exp(
            self.offsets.unsqueeze(0)
            + self.latent_means
            + 1 / 2 * self.latent_sqrt_variances**2
        )
        return torch.sum(exp_term * self.latent_prob.T.unsqueeze(2), dim=0)

    def biplot(
        self,
        column_names,
        *,
        column_index: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
        remove_exog_effect: bool = False,
    ):  # pylint:disable=too-many-arguments
        """
        Visualizes variables using the correlation circle along with the pca transformed samples.
        If the `endog` has been given as a pd.DataFrame, the `column_names` have been stored and
        may be indicated with the `column_names` argument. Else, one should provide the
        indices of variables.

        Parameters
        ----------
        column_names : List[str]
            A list of variable names to visualize.
            If `column_index` is `None`, the variables plotted
            are the ones in `column_names`. If `column_index`
            is not `None`, this only serves as a legend.
            Check the attribute `column_names_endog`.
        column_index : Optional[List[int]], optional keyword-only
            A list of indices corresponding to the variables that should be plotted.
            If `None`, the indices are determined based on `column_names_endog`
            given the `column_names`, by default `None`.
            If not None, should have the same length as `column_names`.
        title : str optional, keyword-only
            An additional title for the plot.
        colors : list, optional, keyword-only
            The labels to color the samples, by default the inferred clusters.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.

        Raises
        ------
        ValueError
            If `column_index`  is not None and
            the length of `column_index` is different
            from the length of `column_names`.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_cluster = 3)
        >>> mixture.fit()
        >>> mixture.biplot(column_names=["MALAT1", "ACTB"])
        >>> mixture.biplot(
        >>>    column_names=["A", "B"],
        >>>    column_index=[1, 3],
        >>>    colors=data["labels"],)


        See also
        --------
        :func:`pyPLNmodels.PlnMixture.viz`
        :func:`pyPLNmodels.PlnMixture.predict_clusters`
        :func:`pyPLNmodels.PlnMixture.pca_pairplot`
        """
        if colors is None:
            colors = self.clusters
        super().biplot(
            column_names=column_names,
            column_index=column_index,
            colors=colors,
            title=title,
            remove_exog_effect=remove_exog_effect,
        )

    @property
    def _clusters(self):
        return torch.argmax(self._latent_prob, dim=1).detach()

    @property
    def clusters(self):
        """
        The predicted clusters of each sample.
        """
        return self._clusters.cpu()

    @property
    def _marginal_means(self):
        if self._exog is None:
            return self._cluster_bias.unsqueeze(1).repeat_interleave(
                self.n_samples, dim=1
            )
        return self._marginal_mean.unsqueeze(0) + self._cluster_bias.unsqueeze(1)

    def compute_elbo(self):
        self._per_sample_per_cluster_elbo = per_sample_elbo_pln_mixture_diag(
            endog=self._endog,
            marginal_means=self._marginal_means,
            offsets=self._offsets,
            latent_means=self._latent_means,
            latent_sqrt_variances=self._latent_sqrt_variances,
            diag_precisions=1 / (self._covariances),
        )
        elbo = torch.sum(self._per_sample_per_cluster_elbo * self._latent_prob.T)
        entropy = torch.sum(torch.xlogy(self._latent_prob, self._weights.unsqueeze(0)))
        entropy -= torch.sum(torch.xlogy(self._latent_prob, self._latent_prob))
        return elbo + entropy

    def _update_closed_forms(self):
        log_latent_prob = (
            torch.log(self._weights.unsqueeze(0)) + self._per_sample_per_cluster_elbo.T
        )
        with torch.no_grad():
            self._latent_prob = torch.nn.Softmax(dim=1)(log_latent_prob)
            self._weights = torch.mean(self._latent_prob, axis=0)

    @property
    @_add_doc(BaseModel)
    def dict_latent_parameters(self):
        """Dictionary of latent parameters."""
        return {
            "latent_prob": self.latent_prob,
            "latent_means": self.latent_means,
            "latent_sqrt_variances": self.latent_sqrt_variances,
            "cluster_bias": self.cluster_bias,
        }

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        """Dictionary of model parameters."""
        return {
            "coef": self.coef,
            "covariances": self.covariances,
            "weights": self.weights,
            "cluster_bias": self.cluster_bias,
        }

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_cluster = 3)
        >>> mixture.fit()
        >>> print("Shape latent positions: ", mixture.latent_positions.shape)
        >>> mixture.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture.latent_variables`
        """,
    )
    def latent_positions(self):
        return self.latent_variables - self.marginal_mean

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_cluster = 3)
        >>> mixture.fit()
        >>> print("Shape latent variables: ", mixture.latent_variables.shape)
        >>> mixture.viz() # Visualize the latent variables
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture.latent_positions`
        """,
    )
    def latent_variables(self):
        """Latent variables."""
        variables = torch.zeros(self.n_samples, self.dim).to(DEVICE)
        clusters = self._clusters
        for k in range(self.n_cluster):
            indices = clusters == k
            variables[indices] += self._latent_means[k, indices]
        # return torch.sum(self.latent_means * (self.latent_prob.T).unsqueeze(2), dim = 0)
        return variables.detach().cpu()

    @property
    def latent_means(self):
        """
        Tensor of latent_means of shape (n_cluster, n_samples, dim).
        Each vector corresponds to the latent_mean of each cluster.
        """
        return self._latent_means.detach().cpu()

    @property
    def latent_sqrt_variances(self):
        """
        Tensor of latent_sqrt_variances of shape (n_cluster, n_samples, dim).
        Each vector corresponds to the latent_sqrt_variance of each cluster.
        """
        return self._latent_sqrt_variances.detach().cpu()

    @property
    def latent_prob(self):
        """
        Latent probability that sample i corresponds to cluster k.
        Returns a torch.Tensor of size (n_samples, n_cluster).
        """
        return self._latent_prob.detach().cpu()

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        """List of parameters needing gradient."""
        list_param = [
            self._latent_means,
            self._latent_sqrt_variances,
            self._sqrt_covariances,
            self._cluster_bias,
        ]
        if self._exog is not None:
            list_param.append(self._coef)
        return list_param

    @property
    def number_of_parameters(self):
        """Number of parameters."""
        return (2 * self.n_cluster + self.nb_cov) * self.dim

    def pca_pairplot(
        self,
        n_components: int = 3,
        colors: np.ndarray = None,
        remove_exog_effect: bool = False,
    ):
        """
        Generates a scatter matrix plot based on Principal
        Component Analysis (PCA) on the latent variables.

        Parameters
        ----------
        n_components : int, optional
            The number of components to consider for plotting.
            Defaults to 3. It cannot be greater than 6.
        colors : np.ndarray, optional
            An array with one label for each sample in the `endog` property of the object.
            Defaults to the inferred clusters.
        remove_exog_effect : bool, optional
            Whether to remove or not the effect of exogenous variables. Defaults to `False`.

        Raises
        ------
        ValueError
            If the number of components requested is greater
            than the number of variables in the dataset.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_cluster=3)
        >>> mixture.fit()
        >>> mixture.pca_pairplot(n_components=5)
        >>> mixture.pca_pairplot(n_components=5, colors=data["labels"])

        See also
        --------
        :func:`pyPLNmodels.PlnMixture.biplot`
        :func:`pyPLNmodels.PlnMixture.predict_clusters`
        :func:`pyPLNmodels.PlnMixture.viz`
        """
        if colors is None:
            colors = self.clusters
        super().pca_pairplot(
            n_components=n_components,
            colors=colors,
            remove_exog_effect=remove_exog_effect,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_cluster = 3)
        >>> mixture.fit()
        >>> mixture.plot_correlation_circle(column_names=["MALAT1", "ACTB"])
        >>> mixture.plot_correlation_circle(column_names=["A", "B"], column_index=[0, 4])
        """,
    )
    def plot_correlation_circle(self, column_names, column_index=None, title: str = ""):
        super().plot_correlation_circle(
            column_names=column_names,
            column_index=column_index,
            title=title,
        )

    @property
    def n_cluster(self):
        """Number of clusters in the model."""
        return self._n_cluster

    @property
    def cluster_bias(self):
        """
        The mean that is associated to each cluster, of size (n_cluster, dim).
        This does not encompass the mean that does not depend on each cluster.
        Each vector cluster_bias[k] is the bias associated to cluster k.
        """
        return self._cluster_bias.detach().cpu()

    def predict_clusters(self, endog: ArrayLike, *, exog=None, offsets=None):
        """
        Predict the clusters of the given endog and exog.
        The dimensions of `endog`, `exog`, and `offsets` should match the ones given in the model.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to `None`.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to `None`.

        Raises
        ------
        ValueError
            If the endog (or exog) has wrong shape compared to the previously fitted
            endog (or exog) variables.
        Returns
        -------
        list: The predicted clusters

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture(data["endog"], n_cluster = 3).fit()
        >>> pred = mixture.predict_clusters(data["endog"])
        >>> print('pred', pred)
        """
        mixture_pred = _PlnMixturePredict(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=self._compute_offsets_method,
            cluster_bias=self._cluster_bias,
            coef=self._coef,
            sqrt_covariances=self._sqrt_covariances,
            weights=self._weights,
        )
        _check_dimensions_for_prediction(
            mixture_pred.endog, self._endog, mixture_pred.exog, self._exog
        )
        mixture_pred.fit()
        return torch.argmax(mixture_pred.latent_prob, dim=1).tolist()

    @property
    def _additional_attributes_list(self):
        return [
            ".covariances",
            ".cluster_bias",
            ".latent_prob",
            ".n_cluster",
            ".weights",
        ]

    @_add_doc(BaseModel)
    def sigma(self):
        return self.covariances

    @property
    def _additional_methods_list(self):
        return [".predict_clusters()"]

    def _get_two_dim_latent_variances(self, sklearn_components):
        latent_variances = torch.zeros(self.n_cluster, self.n_samples, 2, 2)
        for k in range(self.n_cluster):
            latent_variances[k] = torch.from_numpy(
                _get_two_dim_latent_variances(
                    sklearn_components, self.latent_sqrt_variances[k]
                )
            )
        return latent_variances

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return (
            entropy_gaussian_mixture(self._latent_sqrt_variances**2, self._latent_prob)
            .detach()
            .cpu()
            + entropy_clusters(self._latent_prob, self._weights).detach().cpu()
        ).item()

    @property
    def WCSS(self):
        """
        Compute the Within-Cluster Sum of Squares on the latent positions.

        The higher the better, but increasing n_cluster can only increase the
        metric. A trade-off (with the elbow method for example) must be applied.

        Returns positive float.
        """
        return _calculate_wcss(self.latent_positions, self.clusters, self.n_cluster)

    @property
    def silhouette(self):
        """
        Compute the silhouette score on the latent_positions.
        See scikit-learn.metrics.silhouette_score for more information.

        The higher the better.

        Returns float between -1 and 1.
        """
        return silhouette_score(self.latent_positions, self.clusters)


class _PlnMixturePredict(PlnMixture):
    def __init__(
        self,
        *,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        add_const: bool = False,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        cluster_bias: torch.Tensor,
        coef: torch.Tensor,
        sqrt_covariances: torch.Tensor,
        weights: torch.Tensor,
    ):  # pylint: disable=too-many-arguments
        n_cluster = cluster_bias.shape[0]
        self._cluster_bias = cluster_bias.detach()
        if coef is None:
            self._coef = None
        else:
            self._coef = coef.detach()
        self._sqrt_covariances = sqrt_covariances.detach()
        self._weights = weights.detach()
        super().__init__(
            endog=endog,
            n_cluster=n_cluster,
            exog=exog,
            add_const=False,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
        )

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._latent_means, self._latent_sqrt_variances]

    def fit(
        self,
    ):  # pylint: disable=arguments-differ
        return super().fit(maxiter=30, lr=0.01, tol=0, verbose=False)

    def _print_beginning_message(self):
        print("Doing a VE step.")

    def _print_end_of_fitting_message(self, stop_condition=None, tol=None):
        print("Done!")

    def _print_start_init(self):
        pass

    def _print_end_init(self):
        pass

    @property
    def _useful_attributes_list(self):
        return [
            ".latent_variables",
            ".latent_positions",
            ".coef",
            ".model_parameters",
            ".latent_parameters",
            ".optim_details",
        ]
