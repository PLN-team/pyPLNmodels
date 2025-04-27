from typing import Optional, Union

import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.calculations._initialization import _init_coef_coef_inflation
from pyPLNmodels.calculations._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
    _closed_formula_latent_prob,
)
from pyPLNmodels.calculations.elbos import profiled_elbo_zipln
from pyPLNmodels.utils._utils import _add_doc
from pyPLNmodels.utils._viz import _viz_variables, _pca_pairplot, ZIModelViz, _show_prob
from pyPLNmodels.calculations.entropies import entropy_gaussian, entropy_bernoulli
from pyPLNmodels.utils._data_handler import (
    _handle_inflation_data,
    _array2tensor,
    _extract_data_inflation_from_formula,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NULL_TENSOR = torch.tensor([0], device=DEVICE)


class ZIPln(BaseModel):  # pylint: disable=too-many-public-methods
    """
    Zero-Inflated Pln (ZIPln) class. Like a Pln but adds zero-inflation.
    Fitting such a model is slower than fitting a Pln. For more details,
    see Batardière, Chiquet, Gindraud, Mariadassou (2024) “Zero-inflation
    in the Multivariate Poisson Lognormal Family.”

    Examples
    --------
    >>> from pyPLNmodels import ZIPln, Pln, load_microcosm
    >>> data = load_microcosm() # microcosm dataset is highly zero-inflated (96% of zeros)
    >>> zi = ZIPln.from_formula("endog ~ 1 + site", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> # Here Pln is not appropriate:
    >>> pln = Pln.from_formula("endog ~ 1 + site", data)
    >>> pln.fit()
    >>> pln.viz(colors = data["site"])
    >>> # Can also give different covariates:
    >>> zi_diff = ZIPln.from_formula("endog ~ 1 + site | 1 + time", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> ## Or take all the covariates
    >>> zi_all = ZIPln.from_formula("endog ~ 1 + site*time | 1 + site*time", data)
    >>> zi_all.fit()

    See also
    --------
    :func:`pyPLNmodels.ZIPln.from_formula`
    :func:`pyPLNmodels.ZIPln.__init__`
    :class:`pyPLNmodels.ZIPlnPCA`
    """

    _latent_prob: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    _ModelViz = ZIModelViz

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

        Returns
        -------
        A `ZIPln` object

        See also
        --------
        :func:`pyPLNmodels.ZIPln.from_formula`

        Examples
        --------
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> rna = load_scrna()
        >>> zi = ZIPln(rna["endog"], add_const = True)
        >>> zi.fit()
        >>> print(zi)

        """
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        self._exog_inflation, self.column_names_exog_inflation, self._dirac = (
            _handle_inflation_data(exog_inflation, add_const_inflation, self._endog)
        )

    @classmethod
    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPln, load_microcosm
            >>> data = load_microcosm()
            >>> # same covariates for the zero inflation and the gaussian component
            >>> zi_same = ZIPln.from_formula("endog ~ 1 + site", data = data)
            >>> # different covariates
            >>> zi_different = ZIPln.from_formula("endog ~ 1  + site | 1 + time", data = data)
        """,
        returns="""
            `ZIPln`
        """,
        see_also="""
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
    ):
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
        )

    def _init_model_parameters(self):
        _, self._coef_inflation = _init_coef_coef_inflation(
            endog=self._endog,
            exog=self._exog,
            exog_inflation=self._exog_inflation,
            offsets=self._offsets,
        )
        # coef and covariance are not initialized as defined by closed forms.

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1", data)
        >>> zi.fit()
        >>> print(zi)

        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1 | 1 + labels", data)
        >>> zi.fit(maxiter = 500, verbose = True)
        >>> print(zi)
        """,
        returns="""
        ZIPln object
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

    def _init_latent_parameters(self):
        self._latent_mean = torch.log(self._endog + (self._endog == 0)).to(DEVICE)
        self._latent_sqrt_variance = (
            1 / 2 * torch.ones((self.n_samples, self.dim)).to(DEVICE)
        )
        self._latent_prob = torch.sigmoid(self._marginal_mean_inflation) * self._dirac

    @property
    def _covariance(self):
        return _closed_formula_covariance(
            self._marginal_mean,
            self._latent_mean,
            self._latent_sqrt_variance,
            self.n_samples,
        )

    @property
    def _marginal_mean_inflation(self):
        return self._exog_inflation @ self._coef_inflation

    @property
    def latent_prob(self):
        """
        The probabilities that the zero inflation variable is 0.
        """
        return self._latent_prob.detach().cpu()

    def _get_two_dim_latent_variances(self, sklearn_components):
        components_var = np.expand_dims(
            self.latent_sqrt_variance**2, 1
        ) * np.expand_dims(sklearn_components, 0)
        covariances = np.matmul(components_var, np.expand_dims(sklearn_components.T, 0))
        return covariances

    @property
    def _coef(self):
        return _closed_formula_coef(self._exog, self._latent_mean)

    @_add_doc(BaseModel)
    def compute_elbo(self):
        return profiled_elbo_zipln(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            latent_prob=self._latent_prob,
            marginal_mean_inflation=self._marginal_mean_inflation,
            dirac=self._dirac,
        )

    @property
    @_add_doc(BaseModel)
    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_variance**2
        ) * (1 - self.latent_prob)

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        return [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._coef_inflation,
            self._latent_prob,
        ]

    def _project_parameters(self):
        with torch.no_grad():
            self._latent_prob = torch.maximum(
                self._latent_prob,
                NULL_TENSOR,
                out=self._latent_prob,
            )
            self._latent_prob = torch.minimum(
                self._latent_prob,
                1 - NULL_TENSOR,
                out=self._latent_prob,
            )
            self._latent_prob *= self._dirac

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        default = self._default_dict_model_parameters
        return {**default, **{"coef_inflation": self.coef_inflation}}

    @property
    @_add_doc(BaseModel)
    def dict_latent_parameters(self):
        default = self._default_dict_latent_parameters
        return {**default, **{"latent_prob": self.latent_prob}}

    @property
    @_add_doc(BaseModel)
    def latent_variables(self):
        return (
            1 - self.latent_prob
        ) * self.latent_mean + self.marginal_mean * self.latent_prob

    @property
    def latent_prob_variables(self):
        """
        The (conditional) probabilities of the latent probability variables.
        """
        return self.latent_prob

    @_add_doc(
        BaseModel,
        example="""
              >>> from pyPLNmodels import ZIPln, load_microcosm
              >>> data = load_microcosm()
              >>> zi = ZIPln.from_formula("endog ~ 1", data = data)
              >>> zi.fit()
              >>> transformed_endog = zi.transform()
              >>> print(transformed_endog.shape)
              """,
    )
    def transform(self, remove_exog_effect: bool = True):
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + (self.dim + 1) / 2 + self.nb_cov_inflation)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPln, load_microcosm
        >>> data = load_microcosm()
        >>> zi = ZIPln.from_formula("endog ~ 1", data = data)
        >>> zi.fit()
        >>> zi.pca_pairplot(n_components = 5)
        >>> zi.pca_pairplot(n_components = 5, colors = data["time"])
        """,
    )
    def pca_pairplot(
        self, n_components: bool = 3, colors=None, remove_exog_effect: bool = False
    ):
        super().pca_pairplot(
            n_components=n_components,
            colors=colors,
            remove_exog_effect=remove_exog_effect,
        )

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
        """
        min_n_components = min(6, n_components)
        array = self.latent_prob.numpy()
        _pca_pairplot(array, min_n_components, colors)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPln, load_microcosm
        >>> data = load_microcosm()
        >>> zi = ZIPln.from_formula("endog ~ 1", data = data)
        >>> zi.fit()
        >>> zi.plot_correlation_circle(column_names = ["ASV_315", "ASV_749"])
        >>> zi.plot_correlation_circle(column_names = ["A", "B"], column_index = [0,2])
        """,
    )
    def plot_correlation_circle(self, column_names, column_index=None, title: str = ""):
        super().plot_correlation_circle(
            column_names=column_names,
            column_index=column_index,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import ZIPln, load_microcosm
        >>> data = load_microcosm()
        >>> zi = ZIPln.from_formula("endog ~ 1", data = data)
        >>> zi.fit()
        >>> zi.biplot(column_names = ["ASV_315", "ASV_749"])
        >>> zi.biplot(column_names = ["A", "B"], column_index = [0,2], colors = data["time"])
        """,
    )
    def biplot(
        self,
        column_names,
        *,
        column_index: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
        remove_exog_effect: bool = False,
    ):  # pylint:disable=too-many-arguments
        super().biplot(
            column_names=column_names,
            column_index=column_index,
            colors=colors,
            title=title,
            remove_exog_effect=remove_exog_effect,
        )

    @property
    def _additional_attributes_list(self):
        return [".latent_prob"]

    @property
    def _additional_methods_list(self):
        return [
            ".viz_prob()",
            ".show_prob()",
            ".predict_prob_inflation()",
            ".pca_pairplot_prob()",
        ]

    @property
    def _description(self):
        msg = "full covariance."
        return msg

    @property
    def nb_cov_inflation(self):
        """Number of covariates associated with the zero inflation."""
        return self._exog_inflation.shape[1]

    @property
    def exog_inflation(self):
        """
        Property representing the exogenous variables (covariates) associated
        with the zero inflation.

        Returns
        -------
        torch.Tensor
            The exogenous variables of the zero inflation variable.
        """
        return self._exog_inflation.detach().cpu()

    @_array2tensor
    def predict_prob_inflation(
        self, exog_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
    ):
        """
        Method for estimating the probability of a zero coming from the zero-inflated component.

        Parameters
        ----------
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
            The exogenous variables associated with the zero inflation.

        Returns
        -------
        torch.Tensor
            The predicted values `sigmoid(exog_inflation @ coef_inflation)`.

        Raises
        ------
        RuntimeError
            If the shape of the `exog_inflation` is incorrect.
        ValueError
            If the exog inflation given is None.

        Notes
        -----
        - The mean `sigmoid(exog_inflation @ coef_inflation)` is returned.
        - `exog_inflation` should have the shape `(_, nb_cov)`, where `nb_cov` is
          the number of exogenous variables.
        """
        if exog_inflation is None:
            raise ValueError("exog_inflation cannot be None.")
        if exog_inflation.shape[-1] != self.nb_cov_inflation:
            error_string = f"X has wrong shape:({exog_inflation.shape}). Should"
            error_string += f" be (integer, {self.nb_cov_inflation})."
            raise RuntimeError(error_string)
        out = torch.sigmoid(exog_inflation @ self._coef_inflation)
        return out.cpu()

    @property
    def coef_inflation(self):
        """
        Property representing the regression coefficients associated with the zero-inflation
        component, of size (`nb_cov_inflation`, `dim`).

        Returns
        -------
        torch.Tensor
            The coefficients.
        """
        return self._coef_inflation.detach().cpu()

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPln, load_microcosm
            >>> data = load_microcosm()
            >>> zi = ZIPln.from_formula("endog ~ 1 + site", data = data)
            >>> zi.fit()
            >>> zi.viz()
            >>> zi.viz(colors = data["site"])
            >>> zi.viz(show_cov = True)
            >>> zi.viz(remove_exog_effect = True, colors = data["site"])
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
            >>> from pyPLNmodels import ZIPln, load_microcosm
            >>> data = load_microcosm()
            >>> zi = ZIPln.from_formula("endog ~ 1 + site", data = data)
            >>> zi.fit()
            >>> zi.viz_prob()
            >>> zi.viz_prob(colors = data["site"])
        """
        _viz_variables(self.latent_prob, ax=ax, colors=colors, covariances=None)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import ZIPln, load_scrna
            >>> data = load_scrna()
            >>> zi = ZIPln(data["endog"])
            >>> zi.fit()
            >>> zi.plot_expected_vs_true()
            >>> zi.plot_expected_vs_true(colors = data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @property
    @_add_doc(BaseModel)
    def latent_positions(self):
        return self.latent_variables - self.marginal_mean

    @property
    def _closed_latent_prob(self):
        return _closed_formula_latent_prob(
            self._marginal_mean,
            self._offsets,
            self._marginal_mean_inflation,
            self._covariance,
            self._dirac,
        )

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return (
            entropy_gaussian(self._latent_sqrt_variance**2).detach().cpu().item()
            + entropy_bernoulli(self.latent_prob).item()
        )

    def show_prob(self, savefig=False, name_file="", figsize: tuple = (10, 10)):
        """
        Display the model parameters, norm evolution of the parameters and the criterion.

        Parameters
        ----------
        savefig : bool, optional
            If `True`, the figure will be saved to a file. Default is `False`.
        name_file : str, optional
            The name of the file to save the figure. Only used if savefig is `True`.
            Default is an empty string.
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,10)

        """
        _show_prob(
            latent_prob=self.latent_prob,
            column_names_endog=self.column_names_endog,
            savefig=savefig,
            name_file=name_file,
            figsize=figsize,
            model_name=self._name,
        )
