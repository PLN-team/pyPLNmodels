from typing import Optional, Union
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm, t

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.calculations._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
)
from pyPLNmodels.calculations.elbos import profiled_elbo_pln
from pyPLNmodels.calculations.sandwich import SandwichPln
from pyPLNmodels.calculations._initialization import _init_latent_pln
from pyPLNmodels.calculations.entropies import entropy_gaussian
from pyPLNmodels.utils._utils import (
    _add_doc,
    _shouldbefitted,
    _none_if_no_exog,
    _get_two_dim_latent_variances,
)
from pyPLNmodels.utils._viz import _plot_forest_coef


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Pln(BaseModel):  # pylint: disable=too-many-public-methods
    """Simplest model, that is the original PLN model from
    Aitchison, J., and C. H. Ho. “The Multivariate Poisson-Log Normal Distribution.” Biometrika.
    Variance estimation of regression coefficients are available,
    thanks to:
    "Evaluating Parameter Uncertainty in the Poisson Lognormal Model
    with Corrected Variational Estimators" from Batardière, B., Chiquet, J., Mariadassou, M.

        .

    Examples
    --------
    >>> from pyPLNmodels import Pln, load_scrna
    >>> data = load_scrna()
    >>> pln = Pln(data["endog"])
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])

    >>> from pyPLNmodels import Pln, load_scrna
    >>> data = load_scrna()
    >>> pln = Pln.from_formula("endog ~ 1 + labels", data=data)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])


    See also
    --------
    :class:`pyPLNmodels.PlnDiag`
    :class:`pyPLNmodels.PlnPCA`
    :class:`pyPLNmodels.PlnAR`
    :class:`pyPLNmodels.sandwich.SandwichPln`
    :func:`pyPLNmodels.Pln.__init__`
    :func:`pyPLNmodels.Pln.from_formula`
    """

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln(data["endog"])
            >>> pln.fit()
            >>> print(pln)
        """,
        returns="""
            Pln
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.from_formula`
        :class:`pyPLNmodels.PlnPCA`
        :class:`pyPLNmodels.PlnMixture`
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
    ):  # pylint: disable=too-many-arguments
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
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data=data)
        """,
        returns="""
            Pln
        """,
        see_also="""
        :class:`pyPLNmodels.Pln`
        :func:`pyPLNmodels.Pln.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):
        return super().from_formula(
            formula=formula,
            data=data,
            compute_offsets_method=compute_offsets_method,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data)
        >>> pln.fit()
        >>> print(pln)

        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data)
        >>> pln.fit(maxiter=500, verbose=True)
        >>> print(pln)
        """,
        returns="""
        Pln object
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

    def _init_model_parameters(self):
        """The model parameters are profiled in the ELBO, no need to initialize them."""

    def _init_latent_parameters(self):
        self._latent_mean, self._latent_sqrt_variance = _init_latent_pln(self._endog)

    @property
    def _description(self):
        return "full covariance."

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_variance]

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data)
            >>> pln.fit()
            >>> elbo = pln.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        return profiled_elbo_pln(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
        )

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        return self._default_dict_model_parameters

    @property
    @_add_doc(BaseModel)
    def dict_latent_parameters(self):
        return self._default_dict_latent_parameters

    @property
    def _coef(self):
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _covariance(self):
        return _closed_formula_covariance(
            self._marginal_mean,
            self._latent_mean,
            self._latent_sqrt_variance,
            self.n_samples,
        )

    def _get_two_dim_latent_variances(self, sklearn_components):
        return _get_two_dim_latent_variances(
            sklearn_components, self.latent_sqrt_variance
        )

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data)
        >>> pln.fit()
        >>> print(pln.latent_variables.shape)
        >>> pln.viz() # Visualize the latent variables
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.latent_positions`
        """,
    )
    def latent_variables(self):
        return self.latent_mean

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data)
        >>> pln.fit()
        >>> print("Shape latent positions", pln.latent_positions.shape)
        >>> pln.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.latent_variables`
        """,
    )
    def latent_positions(self):
        return self.latent_mean - self.marginal_mean

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        return self.dim * (self.dim + 2 * self.nb_cov + 1) / 2

    @property
    def _additional_attributes_list(self):
        return []

    @property
    def _additional_methods_list(self):
        return [
            ".get_variance_coef()",
            ".get_confidence_interval_coef()",
            ".summary()",
            ".get_coef_p_values()",
            ".plot_regression_forest()",
        ]

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data=data)
        >>> pln.fit()
        >>> pln.plot_correlation_circle(column_names=["MALAT1", "ACTB"])
        >>> pln.plot_correlation_circle(column_names=["A", "B"], column_index=[0, 4])
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
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data=data)
        >>> pln.fit()
        >>> pln.biplot(column_names=["MALAT1", "ACTB"])
        >>> pln.biplot(column_names=["A", "B"], column_index=[0, 4], colors=data["labels"])
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
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data=data)
        >>> pln.fit()
        >>> pln.pca_pairplot(n_components=5)
        >>> pln.pca_pairplot(n_components=5, colors=data["labels"])
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
              >>> from pyPLNmodels import Pln, load_scrna
              >>> data = load_scrna()
              >>> pln = Pln.from_formula("endog ~ 1", data=data)
              >>> pln.fit()
              >>> transformed_endog = pln.transform()
              >>> print(transformed_endog.shape)
              >>> pln.viz()
              >>> transformed_no_exog = pln.transform(remove_exog_effect=True)
              """,
    )
    def transform(self, remove_exog_effect: bool = False):
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_variance**2
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln(data["endog"])
            >>> pln.fit()
            >>> pln.plot_expected_vs_true()
            >>> pln.plot_expected_vs_true(colors=data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1 + labels", data=data)
            >>> pln.fit()
            >>> pln.viz()
            >>> pln.viz(colors=data["labels"])
            >>> pln.viz(show_cov=True)
            >>> pln.viz(remove_exog_effect=True, colors=data["labels"])
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

    @_none_if_no_exog
    @_shouldbefitted
    def get_variance_coef(self):
        """
        Calculate the variance of the regression coefficients using the sandwich estimator.
        Returns None if there are no exogenous variables in the model.

        Returns
        -------
        torch.Tensor
            Variance of the regression coefficients.

        Raises
        ------
        ValueError
            If the number of samples is less than the product of the
            number of covariates and dimensions.

        Examples
        --------
        >>> from pyPLNmodels import Pln, load_scrna
        >>> rna_data = load_scrna()
        >>> pln = Pln(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> variance = pln.get_variance_coef()
        >>> print('variance', variance)

        See also
        --------
        :func:`pyPLNmodels.Pln.summary`
        :func:`pyPLNmodels.Pln.get_coef_p_values`
        :func:`pyPLNmodels.Pln.get_confidence_interval_coef`
        """
        if self.nb_cov == 0:
            print("No exog in the model, so no coefficients. Returning None")
            return None
        if self.nb_cov * self.dim > self.n_samples:
            msg = f"Not enough samples. The number of samples ({self.n_samples}) "
            msg += f"should be greater than nb_cov * dim ({self.nb_cov} *{self.dim}"
            msg += f"={self.nb_cov*self.dim})."
            raise ValueError(msg)
        sandwich_estimator = SandwichPln(self)
        return sandwich_estimator.get_variance_coef()

    @_none_if_no_exog
    @_shouldbefitted
    def get_confidence_interval_coef(self, alpha: float = 0.05):
        """
        Calculate the confidence intervals for the regression coefficients.
        Returns None if there are no exogenous variables in the model.

        Parameters
        ----------
        alpha : float (optional)
            Significance level for the confidence intervals. Defaults to 0.05.

        Returns
        -------
        interval_low, interval_high : Tuple(torch.Tensor, torch.Tensor)
            Lower and upper bounds of the confidence intervals for the coefficients.

        Examples
        --------
        >>> from pyPLNmodels import Pln, load_scrna
        >>> rna_data = load_scrna()
        >>> pln = Pln(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> interval_low, interval_high = pln.get_confidence_interval_coef()

        >>> import torch
        >>> from pyPLNmodels import Pln, PlnSampler
        >>>
        >>> sampler = PlnSampler(n_samples=1500, add_const=False, nb_cov=4)
        >>> endog = sampler.sample() # Sample Pln data.
        >>>
        >>> pln = Pln(endog, exog=sampler.exog, add_const=False)
        >>> pln.fit()
        >>> interval_low, interval_high = pln.get_confidence_interval_coef(alpha=0.05)
        >>> true_coef = sampler.coef
        >>> inside_interval = (true_coef > interval_low) & (true_coef < interval_high)
        >>> print('Should be around 0.95:', torch.mean(inside_interval.float()).item())

        See also
        --------
        :func:`pyPLNmodels.Pln.summary`
        :func:`pyPLNmodels.Pln.get_coef_p_values`
        """
        variance = self.get_variance_coef()
        quantile = norm.ppf(1 - alpha / 2)
        half_length = quantile * torch.sqrt(variance).detach().cpu()
        interval_low = self.coef - half_length
        interval_high = self.coef + half_length
        return interval_low, interval_high

    @_none_if_no_exog
    @_shouldbefitted
    def get_coef_p_values(self):
        """
        Calculate the p-values for the regression coefficients.
        Returns None if there are no exogenous variables in the model.

        Returns
        -------
        p_values : torch.Tensor
            P-values for the regression coefficients.

        Examples
        --------
        >>> from pyPLNmodels import Pln, load_scrna
        >>> rna_data = load_scrna()
        >>> pln = Pln(rna_data["endog"], exog=rna_data["labels_1hot"], add_const=False)
        >>> pln.fit()
        >>> p_values = pln.get_coef_p_values()
        >>> print('P-values: ', p_values)

        See also
        --------
        :func:`pyPLNmodels.Pln.summary`
        :func:`pyPLNmodels.Pln.get_confidence_interval_coef`
        """
        variance = self.get_variance_coef()
        t_stat = self.coef / torch.sqrt(variance.detach().cpu())
        p_values = 2 * (
            1 - t.cdf(torch.abs(t_stat), df=self.n_samples - self.nb_cov * self.dim)
        )
        return p_values

    @_none_if_no_exog
    @_shouldbefitted
    def summary(self):
        """
        Print a summary of the regression coefficients and their p-values for each dimension.
        Returns None if there are no exogenous variabes in the model.

        Examples
        --------
        >>> from pyPLNmodels import Pln, load_scrna
        >>> rna_data = load_scrna()
        >>> pln = Pln(rna_data["endog"], exog = rna_data["labels_1hot"], add_const = False)
        >>> pln.fit()
        >>> pln.summary()

        See also
        --------
        :func:`pyPLNmodels.Pln.get_confidence_interval_coef`
        """
        p_values = self.get_coef_p_values()
        print("Coefficients and p-values per dimension:")
        for dim_index, dim_name in enumerate(self.column_names_endog):
            print(f"\nDimension: {dim_name}")
            print(f"{'Exog Name':<20} {'Coefficient':>15} {'P-value':>15}")
            for coef, p_val, exog_name in zip(
                self.coef[:, dim_index], p_values[:, dim_index], self.column_names_exog
            ):
                p_val = p_val if p_val > 1e-16 else 1e-16
                print(f"{exog_name:<20} {coef.item():>15.6f} {p_val:>15.2g}")

    def plot_regression_forest(self, alpha: float = 0.05, figsize: tuple = (10, 10)):
        """
        Creates a forest plot for regression coefficients with confidence intervals (5%).

        Parameters
        ----------
        alpha: float
            The confidence parameter.
        figsize: tuple
            The size of the figure.
        """
        if self.nb_cov == 0:
            print("No exog in the model, so no coefficients. Returning None")
            return None
        coef_left, coef_right = self.get_confidence_interval_coef(alpha=alpha)
        return _plot_forest_coef(
            coef_left,
            coef_right,
            self.column_names_endog,
            self.column_names_exog,
            figsize=figsize,
        )

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return entropy_gaussian(self._latent_sqrt_variance**2).detach().cpu().item()
