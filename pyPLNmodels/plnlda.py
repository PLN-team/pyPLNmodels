from typing import Optional, Union

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from pyPLNmodels.pln import Pln
from pyPLNmodels._data_handler import (
    _format_clusters,
    _check_dimensions_equal,
    _extract_data_and_clusters_from_formula,
    _check_dimensions_for_prediction,
    _get_dummies,
)
from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels.elbos import profiled_elbo_pln, elbo_pln
from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels._utils import _add_doc
from pyPLNmodels._viz import _viz_lda, _viz_lda_new


class PlnLDA(Pln):
    """
    Supervised Pln model. The classification is based on Linear Discriminant Analysis (LDA).
    This assumes the user have knowledge of clusters, and should be given in the initialization.
    See J. Chiquet, M. Mariadassou, S. Robin: "The Poisson-Lognormal Model as a Versatile Framework
    for the Joint Analysis of Species Abundances" for more information.

    Examples
    --------
    >>> from pyPLNmodels import PlnLDA, load_scrna, get_confusion_matrix, plot_confusion_matrix
    >>> data = load_scrna()
    >>> endog_train, endog_test = data["endog"][:100],data["endog"][100:]
    >>> labels_train, labels_test = data["labels"][:100], data["endog"][100:]
    >>> lda = PlnLDA(endog_train, clusters = labels_train).fit()
    >>> pred_test = lda.predict_clusters(endog_test)
    >>> confusion = get_confusion_matrix(pred_test, labels_test)
    >>> plot_confusion_matrix(confusion_matrix)

    See also
    --------
    :class:`pyPLNmodels.Pln`
    :class:`pyPLNmodels.PlnPCA`
    :class:`pyPLNmodels.PlnMixture`
    :func:`pyPLNmodels.PlnLDA.__init__`
    :func:`pyPLNmodels.PlnLDA.from_formula`
    """

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
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        self.column_names_clusters = (
            clusters.columns if isinstance(clusters, pd.DataFrame) else None
        )
        self._exog_clusters = _format_clusters(clusters)
        if self._exog_clusters is None:
            raise ValueError("You should give clusters.")
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
        >>> lda = PlnLDA.from_formula("endog ~ 1| labels", data).fit()
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
        return self._marginal_mean_clusters + self._marginal_mean

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
    def _coef_and_coef_clusters(self):
        return _closed_formula_coef(self._exog_and_clusters, self._latent_mean)

    @property
    def _coef(self):
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _coef_clusters(self):
        return _closed_formula_coef(self._exog_clusters, self._latent_mean)

    @property
    def _covariance(self):
        return _closed_formula_covariance(
            self._full_marginal_mean,
            self._latent_mean,
            self._latent_sqrt_variance,
            self.n_samples,
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnLDA, load_scrna
            >>> data = load_scrna()
            >>> lda = PlnLDA.from_formula("endog ~ 1 | labels", data)
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
        >>> pln = PlnLDA(endog_train, clusters = clusters_train)
        >>> pln.fit()
        >>> pred = pln.predict_clusters(endog_test)
        >>> print('pred', pred)
        >>> print('true', clusters_test)
        """
        latent_pos = self._ve_step_latent_pos(endog, exog=exog, offsets=offsets)
        clf = self._get_lda_classifier_fitted()
        pred = clf.predict(latent_pos)
        return pred

    def _ve_step_latent_pos(self, endog, *, exog, offsets):
        pln_pred = _PlnPred(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=self._compute_offsets_method,
            fixed_coef=self._coef.detach(),
            fixed_precision=self._precision.detach(),
        )
        _check_dimensions_for_prediction(
            pln_pred.endog, self._endog, pln_pred.exog, self._exog
        )
        pln_pred.fit()
        return pln_pred.latent_positions

    @property
    def _additional_methods_list(self):
        return [".predict_clusters()"]

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
        remove_exog_effect: bool = False,
    ):
        """
        Visualize the endogenous data in the LDA space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.
        colors : list, optional
            Not implemented for PlnLDA.
        show_cov : bool, optional
            Not implemented for PlnLDA.
        remove_exog_effect: bool, optional
            Not implemented for PlnLDA.

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
        """
        if colors is not None:
            raise ValueError("'colors' is not implemented for PlnLDA.")
        if show_cov is not False:
            raise ValueError("'show_cov' is not implemented for PlnLDA.")
        if remove_exog_effect is not False:
            raise ValueError("'show_cov' is not implemented for PlnLDA.")
        _viz_lda(self.latent_mean + self.marginal_mean_clusters, self.clusters, ax=ax)

    def _get_lda_classifier_fitted(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(
            self.latent_mean + self.marginal_mean_clusters,
            self.clusters,
        )
        return clf

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
        >>> ntrain, ntest = 3000, 200
        >>> nb_cov, n_clusters = 1,3
        >>> sampler = PlnLDASampler(
        >>> n_samples=ntrain + ntest, nb_cov=nb_cov, n_clusters=n_clusters, add_const=False,dim=500)
        >>> endog = sampler.sample()
        >>> known_exog = sampler.known_exog
        >>> clusters = sampler.clusters
        >>> endog_train, endog_test = endog[:ntrain], endog[ntrain:]
        >>> known_exog_train, known_exog_test = known_exog[:ntrain], known_exog[ntrain:]
        >>> clusters_train, clusters_test = clusters[:ntrain],clusters[ntrain:]
        >>> lda = PlnLDA(endog_train,
        >>>    clusters = clusters_train, exog = known_exog_train, add_const = False).fit()
        >>> transformed_endog_train = lda.transform()
        >>> print(transformed_endog_train.shape)

        See also
        --------
        :func:`pyPLNmodels.PlnLDA.transform_new`
        :func:`pyPLNmodels.PlnLDA.viz_transformed`
        """
        if remove_exog_effect is not False:
            raise ValueError("'remove_exog_effect' is not implemented for PlnLDA")
        clf = self._get_lda_classifier_fitted()
        return clf.transform(self.latent_mean + self.marginal_mean_clusters)

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
        >>> ntrain, ntest = 3000, 200
        >>> nb_cov, n_clusters = 1,3
        >>> sampler = PlnLDASampler(
        >>> n_samples=ntrain + ntest, nb_cov=nb_cov, n_clusters=n_clusters, add_const=False,dim=500)
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

        See also
        --------
        :func:`pyPLNmodels.PlnLDA.transform`
        :func:`pyPLNmodels.PlnLDA.viz_transformed`
        :func:`pyPLNmodels.PlnLDA.predict_clusters`
        """
        latent_pos = self._ve_step_latent_pos(endog, exog=exog, offsets=offsets)
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
        >>> nb_cov, n_clusters = 1,3
        >>> sampler = PlnLDASampler(
        >>> n_samples=ntrain + ntest, nb_cov=nb_cov, n_clusters=n_clusters, add_const=False,dim=500)
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
        """
        _viz_lda_new(
            X=self.latent_mean + self.marginal_mean_clusters,
            y=self.clusters,
            new_X_transformed=transformed,
            colors=colors,
            ax=ax,
        )


class _PlnPred(Pln):
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: torch.Tensor,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        fixed_coef: torch.Tensor,
        fixed_precision: torch.Tensor,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )
        if self._exog is None:
            self._fixed_marginal_mean = 0
        else:
            self._fixed_marginal_mean = self._exog @ fixed_coef
        self._fixed_precision = fixed_precision

    def compute_elbo(self):
        return elbo_pln(
            endog=self._endog,
            marginal_mean=self._fixed_marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            precision=self._fixed_precision,
        )

    def fit(
        self,
    ):  # pylint: disable=arguments-differ
        return super().fit(maxiter=30, lr=0.01, tol=0, verbose=False)
