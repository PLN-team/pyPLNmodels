from typing import Optional, Union

import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.plndiag import PlnDiag
from pyPLNmodels.elbos import elbo_plnar
from pyPLNmodels._initialization import _init_coef
from pyPLNmodels._utils import _add_doc


class PlnAR(PlnDiag):
    """
    AutoRegressive PLN (PlnAR) model with one step autocorrelation  on the latent variables.
    This basically assumes the latent variable of sample i depends on the latent variable
    on sample i-1. This assumes the dataset given in the initialization is ordered !
    The covariance is assumed diagonal.
    See ?? for more details.

    Examples
    --------
    >>> from pyPLNmodels import PlnAR, load_crossover
    >>> data = load_crossover()
    >>> ar = PlnAR(data["endog"])
    >>> ar.fit()
    >>> print(ar)
    >>> ar.viz(colors=data["labels"])

    >>> from pyPLNmodels import PlnAR, load_crossover
    >>> data = load_crossover()
    >>> ar = PlnAR.from_formula("endog ~ 1 + labels", data=data)
    >>> ar.fit()
    >>> print(ar)
    >>> ar.viz(colors=data["labels"])
    """

    _autoreg_diff_term: torch.Tensor
    __coef: torch.Tensor
    _sqrt_covariance: torch.Tensor

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> pln = PlnAR.from_formula("endog ~ 1", data)
            >>> pln.fit()
            >>> print(pln)
        """,
        returns="""
            PlnAR
        """,
        see_also="""
        :func:`pyPLNmodels.PlnAR.from_formula`
        :class:`pyPLNmodels.PlnDiag`
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
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> pln = PlnAR.from_formula("endog ~ 1", data=data)
        """,
        returns="""
            PlnAR
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
    ):
        return super().from_formula(
            formula=formula,
            data=data,
            compute_offsets_method=compute_offsets_method,
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

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = Pln.from_formula("endog ~ 1", data)
            >>> ar.fit()
            >>> elbo = ar.compute_elbo()
            >>> print(elbo)
        """,
    )
    def compute_elbo(self):
        return elbo_plnar(
            endog=self._endog,
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            covariance=self._covariance,
            ar_matrix=self._autoreg_matrix,
        )

    def _init_model_parameters(self):
        self.__coef = _init_coef(
            endog=self._endog, exog=self._exog, offsets=self._offsets
        )
        self._autoreg_diff_term = torch.ones(self.dim) / 2
        self._sqrt_covariance = torch.ones(self.dim) / 2

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
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
        return super().latent_variables

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
        return super().latent_positions

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
        >>> ar.plot_correlation_circle(variables_names=["MALAT1", "ACTB"])
        >>> ar.plot_correlation_circle(variables_names=["A", "B"], indices_of_variables=[0, 4])
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

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data=data)
        >>> ar.fit()
        >>> ar.biplot(variables_names=["MALAT1", "ACTB"])
        >>> ar.biplot(variables_names=["A", "B"], indices_of_variables=[0, 4], colors=data["labels"])
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
        super().biplot(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            colors=colors,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnAR, load_crossover
        >>> data = load_crossover()
        >>> ar = PlnAR.from_formula("endog ~ 1", data=data)
        >>> ar.fit()
        >>> ar.pca_pairplot(n_components=5)
        >>> ar.pca_pairplot(n_components=5, colors=data["labels"])
        """,
    )
    def pca_pairplot(self, n_components: bool = 3, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
              >>> from pyPLNmodels import PlnAR, load_crossover
              >>> data = load_crossover()
              >>> ar = Pln.from_formula("endog ~ 1", data=data)
              >>> ar.fit()
              >>> print(ar.transform().shape)
              >>> pln.viz()
              >>> transformed_no_exog = pln.transform(remove_exog_effect=True)
              """,
    )
    def transform(self, remove_exog_effect: bool = False):
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    def _endog_predictions(self):
        raise NotImplementedError("Should take into account the time.")

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR(data["endog"])
            >>> ar.fit()
            >>> ar.plot_expected_vs_true()
            >>> ar.plot_expected_vs_true(colors=data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_crossover
            >>> data = load_crossover()
            >>> ar = PlnAR.from_formula("endog ~ 1 + labels", data=data)
            >>> ar.fit()
            >>> ar.viz()
            >>> ar.viz(colors=data["labels"])
            >>> ar.viz(show_cov=True)
            >>> ar.viz(remove_exog_effect=True, colors=data["labels"])
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
        return self._sqrt_covariance**2

    @property
    def _autoreg_matrix(self):
        return 1 / (1 + self._autoreg_diff_term**2)

    @property
    def autoreg_matrix(self):
        """
        Autoregressive model parameters of size p. Defines the correlation
        between sample i and sample i-1 for each dimension.
        The greater the value, the greater the autocorrelation.
        """
        return self._autoreg_matrix.detach().cpu()

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_param = [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._sqrt_covariance,
            self._autoreg_diff_term,
        ]
        if self.__coef is not None:
            return list_param + [self.__coef]
        return list_param

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        return {
            "coef": self.coef,
            "covariance": self.covariance,
            "autoreg_matrix": self.autoreg_matrix,
        }
