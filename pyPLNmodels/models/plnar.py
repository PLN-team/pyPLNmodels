from typing import Optional, Union

import torch
import pandas as pd
import numpy as np

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.calculations.elbos import (
    elbo_plnar_diag_autoreg,
    elbo_plnar_scalar_autoreg,
    smart_elbo_plnar_full_autoreg,
)
from pyPLNmodels.calculations._initialization import (
    _init_coef,
    _init_latent_pln,
    _init_components_prec,
    _init_components,
)
from pyPLNmodels.calculations.entropies import entropy_gaussian
from pyPLNmodels.utils._utils import (
    _add_doc,
    _process_column_index,
    _get_two_dim_latent_variances,
)
from pyPLNmodels.utils._viz import _viz_dims, ARModelViz
from pyPLNmodels.utils._data_handler import _extract_data_from_formula

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnAR(BaseModel):  # pylint: disable=too-many-instance-attributes
    """
    AutoRegressive PLN (PlnAR) model with one step autocorrelation  on the latent variables.
    This basically assumes the latent variable of sample i depends on the latent variable
    on sample i-1. The dataset given in the initialization must be ordered !
    The autoregressive coefficient can be per dimension (`diagonal` ar_type) or
    common to each dimension (`spherical` ar type) or depend on all
    the previous dimensions (`full` ar type). Note that the autregressive coefficient seems
    to be underestimated when the covariance is low.
    See ?? for more details.

    Examples
    --------
    >>> from pyPLNmodels import PlnAR, load_crossover
    >>> data = load_crossover()
    >>> ar = PlnAR(data["endog"])
    >>> ar.fit()
    >>> print(ar)
    >>> ar.viz(colors=data["chrom"])
    >>> ar.viz_dims(column_names = ["nco_Lacaune_M", "nco_Soay_M"])

    >>> from pyPLNmodels import PlnAR, load_crossover
    >>> data = load_crossover()
    >>> ar = PlnAR.from_formula("endog ~ 1 + chrom", data=data)
    >>> ar.fit()
    >>> print(ar)
    >>> ar.viz(colors=data["chrom"])
    >>> ar.viz_dims(column_names = ["nco_Lacaune_F", "nco_Soay_F"])
    """

    _ar_diff_coef: torch.Tensor
    __coef: torch.Tensor
    _sqrt_precision: torch.Tensor
    _components_prec: torch.Tensor
    _diff_ortho_components: torch.Tensor
    _diff_diag_cov: torch.Tensor

    _ModelViz = ARModelViz

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR.from_formula("endog ~ 1", data)
            >>> ar.fit()
            >>> print(ar)
        """,
        params="""
        ar_type: str (optional)
            The autregression type. Can be either "diagonal", "spherical" or "full".
            If "diagonal", the covariance must be diagonal and the model
            boils down to individual independant 1D AR models.
            If "spherical", covariance is full (dependence between variables) but the
            autoregression is shared along the variables as the ar_coef is of size 1.
            If "full", both the covariance and autoregression is full.
            Default is "full".
        """,
        returns="""
            PlnAR
        """,
        see_also="""
        :func:`pyPLNmodels.PlnAR.from_formula`
        :class:`pyPLNmodels.PlnDiag`
        :class:`pyPLNmodels.Pln`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
        ar_type: {"spherical", "diagonal", "full"} = "full",
    ):  # pylint: disable=too-many-arguments
        if ar_type not in ["spherical", "diagonal", "full"]:
            msg = "`ar_type` keyword should be either 'spherical' or 'diagonal' or 'full', got "
            msg += ar_type
            raise AttributeError(msg)
        self._ar_type = ar_type
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> pln = PlnAR.from_formula("endog ~ 1", data=data)
        """,
        returns="""
            PlnAR
        """,
        params="""
        ar_type: str (optional)
            The autregression type. Can be either "diagonal", "spherical" or "full".
            If "diagonal", the covariance must be diagonal and the model
            boils down to individual independant 1D AR models.
            If "spherical", covariance is full (dependence between variables) but the
            autoregression is shared along the variables as the ar_coef is of size 1.
            If "full", both the covariance and autoregression is full.
            Default is "full".
        """,
        see_also="""
        :class:`pyPLNmodels.PlnAR`
        :func:`pyPLNmodels.PlnAR.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        ar_type: {"spherical", "diagonal", "full"} = "full",
    ):
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            ar_type=ar_type,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data)
        >>> ar.fit()
        >>> print(ar)

        >>> from pyPLNmodels import PlnAR, load_scrna
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data)
        >>> ar.fit(maxiter=500, verbose=True)
        >>> print(ar)
        """,
        returns="""
        PlnAR object
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

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR.from_formula("endog ~ 1", data)
            >>> ar.fit()
            >>> elbo = ar.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        if self._ar_type == "diagonal":
            return elbo_plnar_diag_autoreg(
                endog=self._endog,
                marginal_mean=self._marginal_mean,
                offsets=self._offsets,
                latent_mean=self._latent_mean,
                latent_sqrt_variance=self._latent_sqrt_variance,
                precision=self._precision,
                ar_coef=self._ar_coef,
            )
        if self._ar_type == "spherical":
            return elbo_plnar_scalar_autoreg(
                endog=self._endog,
                marginal_mean=self._marginal_mean,
                offsets=self._offsets,
                latent_mean=self._latent_mean,
                latent_sqrt_variance=self._latent_sqrt_variance,
                precision=self._precision,
                ar_coef=self._ar_coef,
            )

        return smart_elbo_plnar_full_autoreg(
            endog=self._endog,
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            ortho_components=self._ortho_components,
            diag_covariance=self._diag_cov,
            diag_ar_coef=self._diag_ar_coef,
        )

    @property
    def _precision(self):
        if self._ar_type == "diagonal":
            return self._sqrt_precision**2
        if self._ar_type == "spherical":
            return self._components_prec @ (self._components_prec.T)
        return torch.inverse(self._covariance)

    def _init_model_parameters(self):
        self.__coef = _init_coef(
            endog=self._endog, exog=self._exog, offsets=self._offsets
        )
        if self.__coef is not None:
            self.__coef = self.__coef.to(DEVICE)
        if self._ar_type == "diagonal":
            self._ar_diff_coef = torch.ones(self.dim).to(DEVICE) / 2
            self._sqrt_precision = torch.ones(self.dim).to(DEVICE) / 2
        elif self._ar_type == "spherical":
            self._components_prec = _init_components_prec(self._endog).to(DEVICE)
            self._ar_diff_coef = torch.tensor([0.5]).to(DEVICE)
        else:
            components = _init_components(self._endog, rank=self.dim).to(DEVICE)
            _, self._diff_ortho_components = torch.linalg.eigh(
                components @ (components.T)
            )
            self._diff_diag_cov = torch.ones(self.dim)
            self._ar_diff_coef = torch.ones(self.dim)

    @property
    def _diag_cov(self):
        return torch.cumsum(self._diff_diag_cov**2, dim=0)

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data)
        >>> ar.fit()
        >>> print("Shape latent positions", ar.latent_positions.shape)
        >>> ar.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
        see_also="""
        :func:`pyPLNmodels.PlnAR.latent_variables`
        """,
    )
    def latent_positions(self):
        return self.latent_variables - self.marginal_mean

    @property
    def _coef(self):
        return self.__coef

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data=data)
        >>> ar.fit()
        >>> ar.plot_correlation_circle(column_names=["nco_Lacaune_M", "nco_Soay_M"])
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
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data=data)
        >>> ar.fit()
        >>> ar.biplot(column_names=["nco_Lacaune_M", "nco_Soay_M"])
        >>> ar.biplot(column_names=["nco_Lacaune_M", "nco_Soay_M"], colors=data["chrom"])
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

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data=data)
        >>> ar.fit()
        >>> ar.pca_pairplot(n_components=3)
        >>> ar.pca_pairplot(n_components=3, colors=data["chrom"])
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

    @_add_doc(
        BaseModel,
        example="""
              >>> from pyPLNmodels import PlnAR, load_crossover
              >>> data = load_crossover()
              >>> ar = PlnAR.from_formula("endog ~ 1 + chrom", data=data)
              >>> ar.fit()
              >>> print(ar.transform().shape)
              >>> ar.viz()
              >>> transformed_no_exog = ar.transform(remove_exog_effect=True)
              """,
    )
    def transform(self, remove_exog_effect: bool = False):
        return super().transform(remove_exog_effect=remove_exog_effect)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR(data["endog"])
            >>> ar.fit()
            >>> ar.plot_expected_vs_true()
            >>> ar.plot_expected_vs_true(colors=data["chrom"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @property
    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_variance**2
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR.from_formula("endog ~ 1 + chrom", data=data)
            >>> ar.fit()
            >>> ar.viz()
            >>> ar.viz(colors=data["chrom"])
            >>> ar.viz(show_cov=True)
            >>> ar.viz(remove_exog_effect=True, colors=data["location"])
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

    @property
    def _covariance(self):
        if self._ar_type == "diagonal":
            return 1 / self._precision
        if self._ar_type == "spherical":
            return torch.inverse(self._precision)
        ortho_components = self._ortho_components
        return ortho_components * self._diag_cov @ (ortho_components.T)

    @property
    def _ortho_components(self):
        ortho_components, _ = torch.linalg.qr(self._diff_ortho_components)
        return ortho_components

    @property
    def _ar_coef(self):
        if self._ar_type != "full":
            return 1 / (1 + self._ar_diff_coef**2)
        ortho_components, _ = torch.linalg.qr(self._diff_ortho_components)
        return ortho_components * self._diag_ar_coef @ (ortho_components.T)

    @property
    def _diag_ar_coef(self):
        return 1 / (1 + self._ar_diff_coef**2)

    @property
    def ar_coef(self):
        """
        Autoregressive model parameters of size p. Defines the correlation
        between sample i and sample i-1 for each dimension.
        The greater the value, the greater the autocorrelation.
        """
        return self._ar_coef.detach().cpu()

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_param = [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._ar_diff_coef,
        ]
        if self._ar_type == "diagonal":
            list_param.append(self._sqrt_precision)
        elif self._ar_type == "spherical":
            list_param.append(self._components_prec)
        else:
            list_param.append(self._diff_ortho_components)
            list_param.append(self._diff_diag_cov)
        if self.__coef is not None:
            return list_param + [self.__coef]
        return list_param

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        return {
            "coef": self.coef,
            "covariance": self.covariance,
            "ar_coef": self.ar_coef,
        }

    def viz_dims(
        self,
        column_names,
        column_index: np.ndarray = None,
        display: {"stretch", "keep"} = "stretch",
        colors: np.ndarray = None,
        *,
        figsize: tuple = (10, 7),
    ):  # pylint: disable=too-many-arguments
        """
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
        display : str (Optional)
            How to display the time series when nan are at stake.
            - "stretch": stretch the time serie so that all time series
              seems to have the same length.
            - "keep": Shorter time series will be displayed shorter.
        colors : list, optional, keyword-only
            The labels to color the samples, of size `n_samples`.
        figsize : tuple of floats.
            The height end width of the matplotlib figsize.
        """
        column_index = _process_column_index(
            column_names, column_index, self.column_names_endog
        )
        if display not in ["stretch", "keep"]:
            msg = "`display` keyword have only two possible values: 'stretch' and 'keep', got"
            msg += str(display)
            raise ValueError(msg)
        _viz_dims(
            variables=self.latent_variables,
            column_index=column_index,
            column_names=column_names,
            colors=colors,
            display=display,
            figsize=figsize,
        )

    @property
    def _additional_methods_list(self):
        return [".viz_dims()"]

    @property
    def _additional_attributes_list(self):
        return [
            ".ar_coef",
        ]

    @property
    def _description(self):
        return f"autoregressive type {self._ar_type}."

    def _init_latent_parameters(self):
        self._latent_mean, self._latent_sqrt_variance = _init_latent_pln(self._endog)

    @property
    def dict_latent_parameters(self):
        return self._default_dict_latent_parameters

    @property
    def number_of_parameters(self):
        if self._ar_type == "diagonal":
            return self.dim * (self.nb_cov + 2)
        if self._ar_type == "spherical":
            return self.dim * (self.dim + 2 * self.nb_cov + 1) / 2 + 1
        return self.dim * (self.dim + self.nb_cov + 1)

    def _get_two_dim_latent_variances(self, sklearn_components):
        return _get_two_dim_latent_variances(
            sklearn_components, self.latent_sqrt_variance
        )

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_scrna
        >>> data = load_scrna()
        >>> ar = PlnAR.from_formula("endog ~ 1", data)
        >>> ar.fit()
        >>> print(ar.latent_variables.shape)
        >>> ar.viz() # Visualize the latent variables
        """,
        see_also="""
        :func:`pyPLNmodels.PlnAR.latent_positions`
        """,
    )
    def latent_variables(self):
        return self.latent_mean

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return entropy_gaussian(self._latent_sqrt_variance**2).detach().cpu().item()
