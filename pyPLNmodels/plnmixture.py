from typing import Optional, Union
import warnings
import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels._data_handler import _extract_data_from_formula
from pyPLNmodels._utils import _add_doc, _two_dim_latent_variances
from pyPLNmodels._initialization import _init_gmm
from pyPLNmodels.elbos import per_sample_elbo_pln_mixture_diag
from pyPLNmodels.plndiag import PlnDiag
from pyPLNmodels._viz import MixtureModelViz


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnMixture(
    BaseModel
):  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """
    Pln mixture models, that is a gaussian mixture model with Poisson layer on top of it.

    Examples
    --------
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from pyPLNmodels import PlnMixture, load_scrna
    >>> data = load_scrna()
    >>> mixture = PlnMixture(data["endog]",n_clusters = 2)
    >>> mixture.fit()
    >>> print(mixture)
    >>> pair_confusion_matrix(mixture.clusters, data["labels"])
    >>> sns.show()


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

    ModelViz = MixtureModelViz

    @_add_doc(
        BaseModel,
        params="""
            n_clusters : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture(data["endog]",n_clusters = 2)
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
        n_clusters: int,
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        add_const: bool = False,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable=too-many-arguments
        if add_const is True:
            msg = "You shall not use the add_const keyword for PlnMixture. Adding "
            msg += "an intercept in the covariates results in non-identifiable coefficients. "
            msg += "This is ignored."
            warnings.warn(msg)
        self._n_clusters = n_clusters
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
            n_clusters : int
                The number of clusters in the model.
            """,
        example="""
            >>> from pyPLNmodels import PlnMixture, load_scrna
            >>> data = load_scrna()
            >>> mixture = PlnMixture.from_formula("endog ~ 1",data, n_clusters = 2)
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
        n_clusters: int,
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable = arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            n_clusters=n_clusters,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 1", data, n_clusters = 2)
        >>> mixture.fit()
        >>> print(mixture)

        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 1", data, n_clusters = 2)
        >>> mixture.fit(maxiter=500, verbose=True)
        >>> print(mixture)
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
        super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    def _covariance(self):
        raise NotImplementedError("No single covariance. Call ._covariances.")

    @property
    def _covariances(self):
        return self._sqrt_covariances**2

    @property
    def covariances(self):
        """
        Tensor of covariances of shape (n_clusters, dim).
        Each vector corresponds to the (diagonal) covariance of each cluster.
        """
        return self._covariances.detach().cpu()

    @property
    def weights(self):
        """
        Probability of a sample to belong to each cluster.
        Tensor of size (n_clusters).
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
            The colors to color the latent variables for the plot, by default the inferred clusters.
        show_cov : bool, optional
            Whether to show covariances, by default False.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_clusters = 2)
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
        if colors is None:
            colors = self.predict_clusters()
        super().viz(
            ax=ax,
            colors=colors,
            show_cov=show_cov,
            remove_exog_effect=remove_exog_effect,
        )

    @property
    def covariance(self):
        """The GMM does not have a single covariance. It has multiple covariances, one per cluster.
        You may call .covariances."""
        raise NotImplementedError(
            "The GMM does not have a single covariance. It has multiple covariances, one"
            " per cluster. You may call .covariances."
        )

    def _init_parameters(self):
        pln = PlnDiag(
            endog=self._endog, exog=self._exog, offsets=self._offsets, add_const=True
        )
        pln.fit(maxiter=150)
        positions_with_mean = pln.latent_positions + (
            pln.exog[:, 0].unsqueeze(1) @ pln.coef[0].unsqueeze(0)
        )

        self._cluster_bias, self._sqrt_covariances, self._weights, self._latent_prob = (
            _init_gmm(positions_with_mean, self.n_clusters)
        )
        if pln.nb_cov > 1:
            self._coef = pln.coef[:-1].to(DEVICE)  # Should not retrieve the mean
        else:
            self._coef = None
        self._latent_means = self._cluster_bias.unsqueeze(1)
        if self._exog is not None:
            self._latent_means = self._latent_means + self._marginal_mean.unsqueeze(0)
        else:
            self._latent_means = self._latent_means.repeat_interleave(
                self.n_samples, dim=1
            )

        self._latent_sqrt_variances = torch.randn(
            self.n_clusters, self.n_samples, self.dim
        ).to(DEVICE)

    def _init_latent_parameters(self):
        """Everything is done in the _init_parameters method."""

    def _init_model_parameters(self):
        """Everything is done in the _init_parameters method."""

    @property
    def _description(self):
        return f"diagonal covariances and {self.n_clusters} clusters."

    @property
    def _endog_predictions(self):
        exp_term = torch.exp(
            self.offsets.unsqueeze(0)
            + self.latent_means
            + 1 / 2 * self.latent_sqrt_variances**2
        )
        return torch.sum(exp_term * self.latent_prob.T.unsqueeze(2), dim=0)

    @_add_doc(
        BaseModel,
        example="""
        """,
        notes="""
        """,
    )
    def biplot(
        self,
        variables_names,
        *,
        indices_of_variables: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
    ):
        """
        Visualizes variables using the correlation circle along with the pca transformed samples.
        If the `endog` has been given as a pd.DataFrame, the `column_names` have been stored and
        may be indicated with the `variables_names` argument. Else, one should provide the
        indices of variables.

        Parameters
        ----------
        variables_names : List[str]
            A list of variable names to visualize.
            If `indices_of_variables` is `None`, the variables plotted
            are the ones in `variables_names`. If `indices_of_variables`
            is not `None`, this only serves as a legend.
        indices_of_variables : Optional[List[int]], optional keyword-only
            A list of indices corresponding to the variables that should be plotted.
            If `None`, the indices are determined based on `column_names_endog`
            given the `variables_names`, by default `None`.
            If not None, should have the same length as `variables_names`.
        title : str optional, keyword-only
            An additional title for the plot.
        colors : list, optional, keyword-only
            The colors to use for the plot, by default the inferred clusters.

        Raises
        ------
        ValueError
            If `indices_of_variables` is None and `column_names_endog` is not set,
            that has been set if the model has been initialized with a pd.DataFrame as `endog`.
        ValueError
            If the length of `indices_of_variables` is different
            from the length of `variables_names`.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture("endog ~ 0", data=data, n_clusters = 2)
        >>> mixture.fit()
        >>> mixture.biplot(variables_names=["MALAT1", "ACTB"])
        >>> mixture.biplot(variables_names=["A", "B"], colors=data["labels"])

        See also
        --------
        :func:`pyPLNmodels.PlnMixture.viz`
        :func:`pyPLNmodels.PlnMixture.predict_clusters`
        :func:`pyPLNmodels.PlnMixture.pca_pairplot`
        """
        if colors is None:
            colors = self.predict_clusters()
        super().biplot(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            colors=colors,
            title=title,
        )

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
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_clusters = 2)
        >>> mixture.fit()
        >>> print("Shape latent positions: ", mixture.latent_positions.shape)
        >>> pln.viz(remove_exog_effect=True) # Visualize the latent positions
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
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data, n_clusters = 2)
        >>> mixture.fit()
        >>> print("Shape latent variables: ", mixture.latent_variables.shape)
        >>> pln.viz() # Visualize the latent variables
        """,
        see_also="""
        :func:`pyPLNmodels.PlnMixture.latent_positions`
        """,
    )
    def latent_variables(self):
        """Latent variables."""
        variables = torch.zeros(self.n_samples, self.dim)
        for k in range(self.n_clusters):
            indices = self.predict_clusters() == k
            variables[indices] += self.latent_means[k, indices]
        # return torch.sum(self.latent_means * (self.latent_prob.T).unsqueeze(2), dim = 0)
        return variables

    @property
    def latent_means(self):
        """
        Tensor of latent_means of shape (n_clusters, n_samples, dim).
        Each vector corresponds to the latent_mean of each cluster.
        """
        return self._latent_means.detach().cpu()

    @property
    def latent_sqrt_variances(self):
        """
        Tensor of latent_sqrt_variances of shape (n_clusters, n_samples, dim).
        Each vector corresponds to the latent_sqrt_variance of each cluster.
        """
        return self._latent_sqrt_variances.detach().cpu()

    @property
    def latent_prob(self):
        """
        Latent probability that sample i corresponds to cluster k.
        Returns a torch.Tensor of size (n_samples, n_clusters).
        """
        return self._latent_prob.detach().cpu()

    @property
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

    def pca_pairplot(self, n_components: bool = 3, colors=None):
        """
        Generates a scatter matrix plot based on Principal
        Component Analysis (PCA) on the latent variables.

        Parameters
        ----------
            n_components (int, optional): The number of components to consider for plotting.
                Defaults to 3. It Cannot be greater than 6.

            colors (np.ndarray): An array with one label for each
                sample in the endog property of the object.
                Defaults to the inferred clusters.

        Raises
        ------
            ValueError: If the number of components requested is greater
                than the number of variables in the dataset.

        Examples
        --------
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = PlnMixture.from_formula("endog ~ 0", data=data, n_clusters = 2)
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
            colors = self.predict_clusters()
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnMixture, load_scrna
        >>> data = load_scrna()
        >>> mixture = Pln.from_formula("endog ~ 0", data=data, n_clusters = 2)
        >>> mixture.fit()
        >>> mixture.plot_correlation_circle(variables_names=["MALAT1", "ACTB"])
        >>> mixture.plot_correlation_circle(variables_names=["A", "B"], indices_of_variables=[0, 4])
        """,
    )
    def plot_correlation_circle(
        self, variables_names, indices_of_variables=None, title: str = ""
    ):
        super().plot_correlation_circle(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            title=title,
        )

    @property
    def n_clusters(self):
        """Number of clusters in the model."""
        return self._n_clusters

    @property
    def cluster_bias(self):
        """
        The mean that is associated to each cluster, of size (n_clusters, dim).
        This does not encompass the mean that does not depend on each cluster.
        Each vector cluster_bias[k] is the bias associated to cluster k.
        """
        return self._cluster_bias.detach().cpu()

    def predict_clusters(self):
        """Predict the clusters of the given endog in the model."""
        return torch.argmax(self.latent_prob, axis=1)

    @property
    def _additional_attributes_list(self):
        return [
            ".clusters",
            ".covariances",
            ".coef",
            ".cluster_bias",
            ".latent_prob",
            ".n_clusters",
            ".weights",
        ]

    @property
    def _additional_methods_list(self):
        return [".predict_clusters()"]

    def _get_two_dim_latent_variances(self, sklearn_components):
        latent_variances = torch.zeros(self.n_clusters, self.n_samples, 2)
        for k in range(self.n_clusters):
            latent_variances[k] = _two_dim_latent_variances(
                sklearn_components, self.latent_sqrt_variances[k]
            )
        return latent_variances
