from typing import Optional, Union

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from pyPLNmodels.models.pln import Pln
from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.utils._data_handler import (
    _format_clusters_and_encoder,
    _check_dimensions_equal,
    _extract_data_and_clusters_from_formula,
    _check_dimensions_for_prediction,
    _get_dummies,
    _check_full_rank_exog,
    _format_data,
)
from pyPLNmodels.calculations._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
)
from pyPLNmodels.calculations.elbos import (
    profiled_elbo_pln,
    per_sample_elbo_pln,
)
from pyPLNmodels.calculations.entropies import entropy_gaussian
from pyPLNmodels.utils._utils import (
    _add_doc,
    _raise_error_1D_viz,
    _process_column_index,
)
from pyPLNmodels.utils._viz import (
    _viz_lda_train,
    _viz_lda_test,
    LDAModelViz,
    _biplot_lda,
    plot_correlation_circle,
)


class PlnLDA(Pln):
    """
    Supervised Pln model. The classification is based on Linear Discriminant Analysis (LDA).
    This assumes the user have knowledge of clusters, and should be given in the initialization.
    See J. Chiquet, M. Mariadassou, S. Robin: "The Poisson-Lognormal Model as a Versatile Framework
    for the Joint Analysis of Species Abundances" for more information.

    Examples
    --------
    >>> from pyPLNmodels import PlnLDA, load_scrna, plot_confusion_matrix
    >>> data = load_scrna()
    >>> endog_train, endog_test = data["endog"][:100],data["endog"][100:]
    >>> labels_train, labels_test = data["labels"][:100], data["labels"][100:]
    >>> lda = PlnLDA(endog_train, clusters = labels_train).fit()
    >>> pred_test = lda.predict_clusters(endog_test)
    >>> plot_confusion_matrix(pred_test, labels_test)

    See also
    --------
    :class:`pyPLNmodels.Pln`
    :class:`pyPLNmodels.PlnPCA`
    :class:`pyPLNmodels.PlnMixture`
    :func:`pyPLNmodels.PlnLDA.__init__`
    :func:`pyPLNmodels.PlnLDA.from_formula`
    """

    _ModelViz = LDAModelViz

    remove_zero_columns = False

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnLDA, load_scrna
            >>> data = load_scrna()
            >>> lda = PlnLDA(data["endog"], clusters = data["labels"])
            >>> lda.fit()
            >>> print(lda)
        """,
        returns="""
            PlnLDA
        """,
        see_also="""
        :func:`pyPLNmodels.PlnLDA.from_formula`
        :class:`pyPLNmodels.Pln`
        :class:`pyPLNmodels.PlnMixture`
        :class:`pyPLNmodels.PlnPCA`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        clusters: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = False,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes the model class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        clusters: Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The known clusters to train on.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to `None`.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to `None`.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overridden (useless) if `offsets` is not None.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the `exog`. Defaults to `False`.
            Will raise an error if False as the exognenous matrix will not be full
            rank if `add_const` is set to `True`, due to clusters being full rank.

        Notes
        -----
        During training, the exogenous variables are composed of
        `exog` and `cluster` being stacked. As a result,
        adding an intercept (`add_const=True`) will result in an error
        as this will result in non-full rank exogenous variables.

        See also
        --------
        :func:`pyPLNmodels.PlnLDA.from_formula`
        :class:`pyPLNmodels.Pln`
        :class:`pyPLNmodels.PlnMixture`
        :class:`pyPLNmodels.PlnPCA`

        Examples
        --------
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> lda = PlnLDA(data["endog"], clusters = data["labels"])
        >>> lda.fit()
        >>> print(lda)

        """
        self.column_names_clusters = (
            clusters.columns if isinstance(clusters, pd.DataFrame) else None
        )
        self._exog_clusters, self._label_encoder = _format_clusters_and_encoder(
            clusters
        )
        if len(self._exog_clusters.shape) == 1:
            self._exog_clusters = _get_dummies(self._exog_clusters)

        if self.column_names_clusters is None:
            self.columna_names_clusters = [
                f"Cluster_{i+1}" for i in range(self._exog_clusters.shape[1])
            ]
        self._compute_offsets_method = compute_offsets_method
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        _check_dimensions_equal(
            "endog", "clusters", self.n_samples, self._exog_clusters.shape[0], 0, 0
        )
        if self._exog is None:
            self._exog_and_clusters = self._exog_clusters
        else:
            self._exog_and_clusters = torch.cat(
                (self._exog, self._exog_clusters), dim=1
            )
        _check_full_rank_exog(self._exog_and_clusters, name_mat="(exog,clusters)")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):
        """
        Create a model instance from a formula and data.

        Parameters
        ----------
        formula : str
            The formula. Must have a pipe '|' after the
            exogenous variables to specify the clusters.
        data : dict
            The data dictionary. Each value can be either a torch.Tensor,
            `np.ndarray`, `pd.DataFrame` or `pd.Series`. The categorical exogenous
            data should be 1-dimensional.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overridden (useless) if data["offsets"] is not `None`.
        Raises
        ------
        ValueError if the formula does not contains the pipe "|".

        Returns
        -------
        PlnLDA object
        """
        endog, exog, offsets, clusters = _extract_data_and_clusters_from_formula(
            formula, data
        )
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            clusters=clusters,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> lda = PlnLDA.from_formula("endog ~ 0| labels", data).fit()
        >>> print(lda)
        """,
        returns="""
        PlnLDA object
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
    def _full_marginal_mean(self):
        """
        Marginal mean that takes both the exog and known clusters as covariates
        """
        return self._exog_and_clusters @ self._coef_and_coef_clusters

    @property
    def clusters(self):
        """
        The clusters of each sample given in initialization.
        """
        return torch.argmax(self._exog_clusters, dim=1).cpu()

    @property
    def _marginal_mean_clusters(self):
        return self._exog_clusters @ (self._coef_clusters)

    @property
    def marginal_mean_clusters(self):
        """
        Marginal mean given only by the clusters mean,
        that is, the mean of each cluster.
        """
        return self._marginal_mean_clusters.cpu().detach()

    @property
    def _covariance(self):
        return _closed_formula_covariance(
            self._full_marginal_mean,
            self._latent_mean,
            self._latent_sqrt_variance,
            self.n_samples,
        )

    @property
    def _coef_and_coef_clusters(self):
        return _closed_formula_coef(self._exog_and_clusters, self._latent_mean)

    @property
    def _coef(self):
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _coef_clusters(self):
        return _closed_formula_coef(self._exog_clusters, self._latent_mean)

    @property
    def coef_clusters(self):
        """Regression coefficients for the cluster variable."""
        return self._coef_clusters.detach().cpu()

    @property
    def dict_model_parameters(self):
        return {
            "coef": self.coef,
            "coef_clusters": self.coef_clusters,
            "covariance": self.covariance,
        }

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnLDA, load_scrna
            >>> data = load_scrna()
            >>> lda = PlnLDA.from_formula("endog ~ 0 | labels", data)
            >>> lda.fit()
            >>> elbo = lda.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        return profiled_elbo_pln(
            endog=self._endog,
            exog=self._exog_and_clusters,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
        )

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
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> endog = data["endog"]
        >>> clusters = data["labels_1hot"]
        >>> n_train, n_test = 100, 100
        >>> endog_train = endog[:n_train]
        >>> endog_test = endog[n_train:]
        >>> clusters_train = clusters[:n_train]
        >>> clusters_test = clusters[n_train:]
        >>> lda = PlnLDA(endog_train, clusters = clusters_train)
        >>> lda.fit()
        >>> pred = lda.predict_clusters(endog_test)
        >>> print('pred', pred)
        >>> print('true', clusters_test)
        """
        prob, _ = self._estimate_prob_and_latent_positions(
            endog, exog=exog, offsets=offsets
        )
        pred = torch.argmax(prob, dim=1)
        if self._label_encoder is None:
            return pred
        return self._label_encoder.inverse_transform(pred)

    def _estimate_prob_and_latent_positions(self, endog, *, exog, offsets):
        endog = _format_data(endog)
        best_guess_gaussian = torch.zeros(endog.shape).to(self._endog.device)
        predicted_prob = torch.zeros((endog.shape[0], self._n_cluster)).to(
            self._endog.device
        )
        best_prob = torch.zeros(endog.shape[0]).to(self._endog.device) - torch.inf
        coef = self._coef.detach() if self._coef is not None else None
        for k in range(self._n_cluster):
            pln_pred = _PlnPred(
                endog=endog,
                exog=exog,
                offsets=offsets,
                compute_offsets_method=self._compute_offsets_method,
                coef_k=self._coef_clusters[k].detach(),
                known_coef=coef,
                fixed_precision=self._precision.detach(),
            )
            _check_dimensions_for_prediction(
                pln_pred.endog, self._endog, pln_pred.exog, self._exog
            )
            pln_pred.fit()
            predicted_prob[:, k] = pln_pred.per_sample_elbo
            better_individuals = predicted_prob[:, k] > best_prob
            best_prob[better_individuals] = predicted_prob[better_individuals, k]
            best_guess_gaussian[better_individuals] = pln_pred.latent_positions_device[
                better_individuals
            ]
        return predicted_prob.detach().cpu(), best_guess_gaussian.detach().cpu()

    @property
    def _additional_methods_list(self):
        return [".predict_clusters()", ".transform_new()", ".viz_transformed()"]

    @property
    def _additional_attributes_list(self):
        return [".clusters", ".latent_positions_clusters"]

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnLDA, load_scrna
            >>> data = load_scrna()
            >>> endog = data["endog"]
            >>> clusters = data["labels_1hot"]
            >>> n_train, n_test = 100, 100
            >>> endog_train = endog[:n_train]
            >>> endog_test = endog[n_train:]
            >>> clusters_train = clusters[:n_train]
            >>> clusters_test = clusters[n_train:]
            >>> lda = PlnLDA(endog_train, clusters = clusters_train).fit()
            >>> lda.viz()
            """,
    )
    def viz(
        self,
        *,
        ax=None,
        colors=None,
        show_cov: bool = False,
        remove_exog_effect: bool = True,
    ):
        """
        Visualize the endogenous data in the LDA space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.
        colors : list, optional
            The labels to color the samples, of size n_samples.
        show_cov : bool, optional
            Not implemented for PlnLDA.
        remove_exog_effect: bool, optional
            Not implemented for PlnLDA. By default, True.
            the effect of exogenous variables is removed.

        Examples
        --------
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> endog = data["endog"]
        >>> clusters = data["labels_1hot"]
        >>> n_train, n_test = 100, 100
        >>> endog_train = endog[:n_train]
        >>> endog_test = endog[n_train:]
        >>> clusters_train = clusters[:n_train]
        >>> clusters_test = clusters[n_train:]
        >>> lda = PlnLDA(endog_train, clusters = clusters_train).fit()
        >>> lda.viz()

        Notes
        -----
        The visualization is different when there are 2 clusters or
        strictly more than 2 clusters. If 2 clusters, visualization
        is only possible in 1D. A random noise is added on the y axis for visualization
        purposes.
        """
        if colors is None:
            colors = self._decode_clusters(self.clusters)
        if show_cov is not False:
            raise ValueError("'show_cov' is not implemented for PlnLDA.")
        if remove_exog_effect is not True:
            raise ValueError("'show_cov' is not implemented for PlnLDA.")
        _viz_lda_train(self.latent_positions_clusters, colors, ax=ax)

    def _get_lda_classifier_fitted(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(
            self.latent_positions_clusters,
            self.clusters,
        )
        return clf

    def _decode_clusters(self, clusters):
        if self._label_encoder is None:
            return clusters
        return self._label_encoder.inverse_transform(clusters)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnLDA, load_scrna
            >>> data = load_scrna()
            >>> lda = PlnLDA(data["endog"], clusters = data["labels"])
            >>> lda.fit()
            >>> lda.plot_expected_vs_true()
            >>> lda.plot_expected_vs_true(colors=data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    def transform(self, remove_exog_effect=False):
        """
        Transform the endog into the learned LDA space. The remove_exog_effect is not
        implemented for PlnLDA.

        Returns
        -------
        torch.Tensor: The transformed data

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
        >>> transformed_endog_train = lda.transform()
        >>> print('shape', transformed_endog_train.shape)

        See also
        --------
        :func:`pyPLNmodels.PlnLDA.transform_new`
        :func:`pyPLNmodels.PlnLDA.viz_transformed`
        """
        if remove_exog_effect is not False:
            raise ValueError("'remove_exog_effect' is not implemented for PlnLDA")
        clf = self._get_lda_classifier_fitted()
        return clf.transform(self.latent_positions_clusters)

    def transform_new(self, endog, *, exog=None, offsets=None):
        """
        Transform the (unseen) endog data into the previously learned LDA space.

        Returns
        -------
        torch.Tensor: The transformed data

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
        >>> transformed_endog_test = lda.transform_new(endog_test, exog = known_exog_test)
        >>> print(transformed_endog_test.shape)

        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna(n_samples = 500)
        >>> n_train = 250
        >>> endog_train, endog_test = data["endog"][:n_train],data["endog"][n_train:]
        >>> labels_train, labels_test = data["labels"][:n_train], data["labels"][n_train:]
        >>> lda = PlnLDA(endog_train, clusters = labels_train).fit()
        >>> endog_test_transformed = lda.transform_new(endog_test)

        See also
        --------
        :func:`pyPLNmodels.PlnLDA.transform`
        :func:`pyPLNmodels.PlnLDA.viz_transformed`
        :func:`pyPLNmodels.PlnLDA.predict_clusters`
        """
        _, latent_pos = self._estimate_prob_and_latent_positions(
            endog, exog=exog, offsets=offsets
        )
        clf = self._get_lda_classifier_fitted()
        return clf.transform(latent_pos)

    def viz_transformed(self, transformed, colors=None, ax=None):
        """
        Visualize the transformed data in the LDA space.

        Parameters
        ----------
        transformed : torch.Tensor
            The transformed data.
        colors : list, optional
            The labels to color the samples, of size transformed.shape[0].
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.

        Examples
        --------
        >>> import torch
        >>> from pyPLNmodels import PlnLDA, PlnLDASampler
        >>> ntrain, ntest = 3000, 200
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
        >>> transformed_endog_test = lda.transform_new(endog_test, exog = known_exog_test)
        >>> lda.viz_transformed(transformed_endog_test)

        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna(n_samples = 500)
        >>> n_train = 250
        >>> endog_train, endog_test = data["endog"][:n_train],data["endog"][n_train:]
        >>> labels_train, labels_test = data["labels"][:n_train], data["labels"][n_train:]
        >>> lda = PlnLDA(endog_train, clusters = labels_train).fit()
        >>> endog_test_transformed = lda.transform_new(endog_test)
        >>> lda.viz_transformed(endog_test_transformed)
        """
        _viz_lda_test(
            transformed_train=self.latent_positions_clusters,
            y_train=self.clusters,
            new_X_transformed=transformed,
            colors=colors,
            ax=ax,
        )

    @property
    def _latent_positions_clusters(self):
        return self._latent_mean - self._marginal_mean

    @property
    def latent_positions_clusters(self):
        """
        The latent positions with effects coming only from the clusters
        covariates (effects of 'known' covariates remove).
        """
        return self._latent_positions_clusters.detach().cpu()

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> pln = PlnLDA.from_formula("endog ~ 0 | labels", data)
        >>> pln.fit()
        >>> print(pln.latent_variables.shape)
        >>> pln.viz() # Visualize the latent variables without exogenous effects.
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.latent_positions`
        """,
    )
    def latent_variables(self):
        return self.latent_mean

    def pca_pairplot(
        self,
        n_components: int = 3,
        colors: np.ndarray = None,
        remove_exog_effect: bool = False,
    ):
        raise NotImplementedError("pca pairplot not implemented for LDA models.")

    @property
    def _n_cluster(self):
        return self._exog_clusters.shape[1]

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> lda = PlnLDA.from_formula("endog ~ 0|labels", data=data)
        >>> lda.fit()
        >>> lda.plot_correlation_circle(column_names=["MALAT1", "ACTB"])
        >>> lda.plot_correlation_circle(column_names=["A", "B"], column_index=[0, 4])
        """,
        raises="""
        ValueError
            If the number of clusters is 2, as the latent variables will be of dimension
            and visualization is not possible.
        """,
    )
    def plot_correlation_circle(self, column_names, column_index=None, title: str = ""):
        if self._n_cluster == 2:
            _raise_error_1D_viz()
        column_index = _process_column_index(
            column_names, column_index, self.column_names_endog
        )
        data_matrix = torch.cat(
            (
                self.latent_positions_clusters,
                self.clusters.unsqueeze(1),
            ),
            dim=1,
        )
        plot_correlation_circle(
            data_matrix=data_matrix,
            column_names=column_names,
            column_index=column_index,
            title=title,
            reduction="LDA",
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnLDA, load_scrna
        >>> data = load_scrna()
        >>> lda = PlnLDA.from_formula("endog ~ 0 | labels", data=data)
        >>> lda.fit()
        >>> lda.biplot(column_names=["MALAT1", "ACTB"])
        >>> lda.biplot(column_names=["A", "B"], column_index=[0, 4], colors=data["labels"])
        """,
        raises="""
        ValueError
            If the number of clusters is 2, as the latent variables will be of dimension
            and visualization is not possible.
        """,
        notes="""
        The effect of covariates is always removed for visualization_purposes.
        """,
    )
    def biplot(
        self,
        column_names,
        *,
        column_index: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
    ):  # pylint: disable=arguments-differ
        if self._n_cluster == 2:
            _raise_error_1D_viz()
        column_index = _process_column_index(
            column_names, column_index, self.column_names_endog
        )
        _biplot_lda(
            self.latent_positions_clusters,
            column_names,
            clusters=self.clusters,
            colors=self._decode_clusters(self.clusters),
            column_index=column_index,
            title=title,
        )


class _PlnPred(Pln):

    per_sample_elbo: torch.Tensor

    remove_zero_columns = False

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: torch.Tensor,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        coef_k: torch.Tensor,
        known_coef: torch.Tensor,
        fixed_precision: torch.Tensor,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )
        self._coef_k = coef_k
        self._known_coef = known_coef
        self._fixed_precision = fixed_precision

    def compute_elbo(self):
        self.per_sample_elbo = per_sample_elbo_pln(
            endog=self._endog,
            marginal_mean=self._fixed_marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            precision=self._fixed_precision,
        )
        return torch.sum(self.per_sample_elbo)

    @property
    def _maginal_mean(self):
        if self._exog is None:
            return 0
        return self._exog @ self._known_coef

    @property
    def _fixed_marginal_mean(self):
        if self._exog is None:
            return self._coef_k.unsqueeze(0).repeat_interleave(self.n_samples, dim=0)
        return self._marginal_mean + self._coef_k.unsqueeze(0)

    def fit(
        self,
    ):  # pylint: disable=arguments-differ
        return super().fit(maxiter=50, lr=0.01, tol=0, verbose=False)

    def _print_beginning_message(self):
        print("Doing a VE step.")

    def _print_end_of_fitting_message(self, stop_condition=None, tol=None):
        print("Done!")

    def _print_start_init(self):
        pass

    def _print_end_init(self):
        pass

    @property
    def latent_positions_device(self):
        """Latent positions on the GPU device if GPU is available."""
        return self._latent_mean - self._marginal_mean

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return entropy_gaussian(self.latent_variance).item()
