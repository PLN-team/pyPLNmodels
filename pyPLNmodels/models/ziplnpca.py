from typing import Optional, Union
import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.zipln import ZIPln
from pyPLNmodels.calculations.elbos import elbo_ziplnpca
from pyPLNmodels.calculations._initialization import (
    _init_coef_coef_inflation,
    _init_components,
    _init_latent_sqrt_variance_pca,
    _init_latent_mean_pca,
)
from pyPLNmodels.utils._data_handler import _extract_data_inflation_from_formula
from pyPLNmodels.utils._utils import _add_doc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ZIPlnPCA(ZIPln):  # pylint: disable= too-many-instance-attributes
    """
    Zero-Inflated Pln Principal Component Analysis (ZIPlnPCA) class.
    Like a PlnPCA but adds zero-inflation. For more details,
    see Bricout, Barbara, Robin, Donnet (2024) ??

    Examples
    --------
    >>> from pyPLNmodels import ZIPlnPCA, PlnPCA, load_microcosm
    >>> data = load_microcosm() # microcosm dataset is highly zero-inflated (96% of zeros)
    >>> zipca = ZIPlnPCA.from_formula("endog ~ 1 + site", data, rank = 5)
    >>> zipca.fit()
    >>> zipca.viz(colors = data["site"])
    >>> # Here PlnPCA is not appropriate:
    >>> pca = PlnPCA.from_formula("endog ~ 1 + site", data)
    >>> pca.fit()
    >>> pca.viz(colors = data["site"])
    >>> # Can also give different covariates:
    >>> zipca_diff = ZIPlnPCA.from_formula("endog ~ 1 + site | 1 + time", data, rank = 5)
    >>> zipca_diff.fit()
    >>> zipca_diff.viz(colors = data["site"])
    >>> ## Or take all the covariates
    >>> zipca_all = ZIPlnPCA.from_formula("endog ~ 1 + site*time | 1 + site*time", data, rank = 5)
    >>> zipca_all.fit()

    See also
    --------
    :func:`pyPLNmodels.ZIPlnPCA.from_formula`
    :func:`pyPLNmodels.ZIPlnPCA.__init__`
    :class:`pyPLNmodels.ZIPln`
    :class:`pyPLNmodels.PlnPCA`
    """

    __coef: torch.Tensor
    _components: torch.Tensor
    _latent_prob: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    def __init__(
        self,
        endog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        exog_inflation: Optional[
            Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
        ] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
        add_const_inflation: bool = True,
        rank: int = 5,
        use_closed_form_prob: bool = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes the ZIPln class, which is a Pln model with zero-inflation.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series], optional(keyword-only)
            The covariate data. Defaults to `None`.
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data for the inflation part. Defaults to `None`.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to `None`.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overridden (useless) if `offsets` is not `None`.
        add_const : bool, optional(keyword-only)
            Whether to add a column of ones in the `exog`. Defaults to `True`.
        add_const_inflation : bool, optional(keyword-only)
            Whether to add a column of ones in the `exog_inflation`. Defaults to `True`.
        rank : int, optional(keyword-only)
            The rank of the approximation, by default 5.
        use_closed_form_prob: bool, optional (keyword-only)
            Weither to use or not the anayltic formula for the latent probability. Default
            is True.


        Returns
        -------
        A `ZIPlnPCA` object

        See also
        --------
        :func:`pyPLNmodels.ZIPlnPCA.from_formula`

        Examples
        --------
        >>> from pyPLNmodels import ZIPlnPCA, load_scrna
        >>> rna = load_scrna()
        >>> zi = ZIPlnPCA(rna["endog"], add_const = True)
        >>> zi.fit()
        >>> print(zi)

        """
        self._rank = rank
        self._use_closed_form_prob = use_closed_form_prob
        super().__init__(
            endog,
            exog=exog,
            exog_inflation=exog_inflation,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
            add_const_inflation=add_const_inflation,
        )

    @property
    def rank(self):
        """Rank of the covariance of the Gaussian latent variable."""
        return self._rank

    @classmethod
    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
            >>> data = load_microcosm()
            >>> # same covariates for the zero inflation and the gaussian component
            >>> zipca_same = ZIPlnPCA.from_formula("endog ~ 1 + site", data = data)
            >>> # different covariates
            >>> zipca_different = ZIPlnPCA.from_formula("endog ~ 1  + site | 1 + time", data = data)
        """,
        params="""
            rank : int, optional(keyword-only)
                The rank of the approximation, by default 5.
            """,
        returns="""
            `ZIPlnPCA`
        """,
        see_also="""
        :class:`pyPLNmodels.ZIPlnPCA`
        :class:`pyPLNmodels.PlnPCA`
        :class:`pyPLNmodels.ZIPln`
        :func:`pyPLNmodels.ZIPln.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        rank: int = 5,
        use_closed_form_prob: bool = True,
    ):  # pylint: disable= too-many-arguments
        endog, exog, offsets, exog_inflation = _extract_data_inflation_from_formula(
            formula, data
        )
        return cls(
            endog,
            exog=exog,
            exog_inflation=exog_inflation,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
            add_const_inflation=False,
            rank=rank,
            use_closed_form_prob=use_closed_form_prob,
        )

    def _init_model_parameters(self):
        self.__coef, self._coef_inflation = _init_coef_coef_inflation(
            endog=self._endog,
            exog=self._exog,
            exog_inflation=self._exog_inflation,
            offsets=self._offsets,
        )
        self._components = _init_components(self._endog, self.rank).to(DEVICE)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPlnPCA, load_scrna
        >>> data = load_scrna()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data)
        >>> zipca.fit()
        >>> print(zipca)

        >>> from pyPLNmodels import ZIPlnPCA, load_scrna
        >>> data = load_scrna()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1 | 1 + labels", data)
        >>> zipca.fit(maxiter = 500, verbose = True)
        >>> print(zipca)
        """,
        returns="""
        ZIPlnPCA object
        """,
    )
    def fit(
        self,
        *,
        maxiter: int = 1000,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL / 1000,
        verbose: bool = False,
    ):

        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    def _init_latent_parameters(self):
        self._latent_mean = _init_latent_mean_pca(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            coef=self.__coef,
            components=self._components,
        )
        self._latent_sqrt_variance = _init_latent_sqrt_variance_pca(
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            components=self._components,
            mode=self._latent_mean,
        )
        self._latent_prob = self._closed_latent_prob

    @property
    def _description(self):
        return f"{self.rank} principal components."

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_params = [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._coef_inflation,
            self._components,
        ]
        if self._use_closed_form_prob is False:
            list_params.append(self._latent_prob)
        if self.__coef is not None:
            return list_params + [self.__coef]
        return list_params

    @property
    def _coef(self):
        # _coef is a method in the ZIPln class. Any cleaner way would be appreciate.
        return self.__coef

    @property
    @_add_doc(BaseModel)
    def latent_variables(self):
        return (1 - self.latent_prob) * torch.matmul(
            self.latent_mean, self.components.T
        ) + self.marginal_mean * self.latent_prob

    @property
    def components(self):
        """
        Returns the principal components of the PlnPCA model, i.e. the weights
        of features that explain the most variance in the data.

        Returns
        -------
        torch.Tensor
            The components with size (dim, rank)
        """
        return self._components.detach().cpu()

    @property
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + self.rank + self.nb_cov_inflation)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
        >>> data = load_microcosm()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data = data)
        >>> zipca.fit()
        >>> zipca.pca_pairplot(n_components = 5)
        >>> zipca.pca_pairplot(n_components = 5, colors = data["time"])
        """,
    )
    def pca_pairplot(self, n_components: bool = 3, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    def pca_pairplot_prob(self, n_components: int = 3, colors: np.ndarray = None):
        """
        Generates a scatter matrix plot based on Principal
        Component Analysis (PCA) on the latent variables associated
        with the zero inflation (i.e. the Bernoulli variables).
        This may not be very informative.

        Parameters
        ----------
        n_components: int (optional)
            The number of components to consider for plotting.
            Defaults to 3. Cannot be greater than 6.
        colors: np.ndarray (optional)
            An array with one label for each
            sample in the endog property of the object. Defaults to `None`.
        See also
        --------
        :func:`pyPLNmodels.ZIPln.pca_pairplot`
        Examples
        --------
        >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
        >>> data = load_microcosm()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data = data)
        >>> zipca.fit()
        >>> zipca.pca_pairplot_prob(n_components = 5)
        >>> zipca.pca_pairplot_prob(n_components = 5, colors = data["time"])
        """
        return super().pca_pairplot_prob(n_components=n_components, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
        >>> data = load_microcosm()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data = data)
        >>> zipca.fit()
        >>> zipca.plot_correlation_circle(variable_names = ["ASV_315", "ASV_749"])
        >>> zipca.plot_correlation_circle(variable_names = ["A", "B"], indices_of_variables = [0,2])
        """,
    )
    def plot_correlation_circle(
        self, variable_names, indices_of_variables=None, title: str = ""
    ):
        super().plot_correlation_circle(
            variable_names=variable_names,
            indices_of_variables=indices_of_variables,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
        >>> data = load_microcosm()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data = data)
        >>> zipca.fit()
        >>> zipca.biplot(variable_names = ["ASV_315", "ASV_749"])
        >>> zipca.biplot(variable_names = ["A", "B"], indices_of_variables = [0,2], colors = data["time"])
        """,
    )
    def biplot(
        self,
        variable_names,
        *,
        indices_of_variables: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
    ):
        super().biplot(
            variable_names=variable_names,
            indices_of_variables=indices_of_variables,
            colors=colors,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
            >>> data = load_microcosm()
            >>> zipca = ZIPlnPCA.from_formula("endog ~ 1 + site", data = data)
            >>> zipca.fit()
            >>> zipca.viz()
            >>> zipca.viz(colors = data["site"])
            >>> zipca.viz(show_cov = True)
            >>> zipca.viz(remove_exog_effect = True, colors = data["site"])
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
        super().viz(
            ax=ax,
            colors=colors,
            show_cov=show_cov,
            remove_exog_effect=remove_exog_effect,
        )

    def viz_prob(self, *, ax=None, colors=None):
        """
        Visualize the latent variables.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.
        colors : list, optional
            The labels to color the samples, of size `n_samples`.

        Examples
        --------
            >>> from pyPLNmodels import ZIPlnPCA, load_microcosm
            >>> data = load_microcosm()
            >>> zipca = ZIPlnPCA.from_formula("endog ~ 1 + site", data = data)
            >>> zipca.fit()
            >>> zipca.viz_prob()
            >>> zipca.viz_prob(colors = data["site"])
        """
        # pylint: disable=useless-parent-delegation
        super().viz_prob(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPlnPCA, load_scrna
            >>> data = load_scrna()
            >>> zipca = ZIPlnPCA(data["endog"])
            >>> zipca.fit()
            >>> zipca.plot_expected_vs_true()
            >>> zipca.plot_expected_vs_true(colors = data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPlnPCA, load_scrna
            >>> data = load_scrna()
            >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data)
            >>> zipca.fit()
            >>> elbo = zipca.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        if self._use_closed_form_prob is True:
            self._latent_prob = self._closed_latent_prob
        return elbo_ziplnpca(
            endog=self._endog,
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            latent_prob=self._latent_prob,
            components=self._components,
            marginal_mean_inflation=self._marginal_mean_inflation,
            dirac=self._dirac,
        )

    @property
    def _covariance(self):
        return self._components @ self._components.T

    def _get_two_dim_latent_variances(self, sklearn_components):
        transformed_components = sklearn_components @ self.components.numpy()
        transformed_components_latent_var = (
            transformed_components[np.newaxis, :]
            * self.latent_variance.numpy()[:, np.newaxis]
        )
        covariances = (
            transformed_components_latent_var @ transformed_components.T[np.newaxis, :]
        )
        return covariances

    @_add_doc(
        BaseModel,
        params="""
        project : bool, optional
            Whether to project the latent variables onto the `rank` first PCs, by default `False`.
        """,
        example="""
            >>> from pyPLNmodels import ZIPlnPCA, load_scrna
            >>> data = load_scrna()
            >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data)
            >>> zipca.fit()
            >>> transformed_endog_low_dim = zipca.transform()
            >>> transformed_endog_high_dim = zipca.transform(project=False)
            >>> print(transformed_endog_low_dim.shape)
            >>> print(transformed_endog_high_dim.shape)
            >>> transformed_no_exog = zipca.transform(remove_exog_effect=True, project=True)
        """,
    )
    def transform(self, remove_exog_effect: bool = False, project=False):
        if project is True:
            return self.projected_latent_variables(
                rank=self.rank, remove_exog_effect=remove_exog_effect
            )
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPlnPCA, load_scrna
        >>> data = load_scrna()
        >>> zipca = ZIPlnPCA.from_formula("endog ~ 1", data)
        >>> zipca.fit()
        >>> print("Shape latent positions:", zipca.latent_positions.shape)
        >>> zipca.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
    )
    def latent_positions(self):
        return super().latent_positions

    @property
    def _endog_predictions(self):
        covariance_a_posteriori = torch.sum(
            (self.components**2).unsqueeze(0)
            * (self.latent_sqrt_variance**2).unsqueeze(1),
            axis=2,
        )
        return torch.exp(
            self.offsets + self.latent_variables + 1 / 2 * covariance_a_posteriori
        ) * (1 - self.latent_prob)

    def _project_parameters(self):
        if self._use_closed_form_prob is False:
            super()._project_parameters()
