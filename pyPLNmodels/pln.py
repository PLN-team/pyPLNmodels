from typing import Optional, Union
import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels.elbos import profiled_elbo_pln
from pyPLNmodels._utils import _add_doc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Pln(BaseModel):
    """Simplest model, that is the original PLN model from
    Aitchison, J., and C. H. Ho. “The Multivariate Poisson-Log Normal Distribution.” Biometrika.

    Examples
    --------
    >>> from pyPLNmodels import Pln, load_scrna
    >>> data = load_scrna()
    >>> pln = Pln(data["endog"],add_const = True)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors = data["labels"])

    >>> from pyPLNmodels import Pln, load_scrna
    >>> data = load_scrna()
    >>> pln = Pln.from_formula("endog ~ 1 + labels", data = data)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors = data["labels"])

    """

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data)
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
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
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
            >>> pln = Pln.from_formula("endog ~ 1", data = data)
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
        >>> pln.fit( maxiter = 500, verbose = True)
        >>> print(pln)
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

    def _init_model_parameters(self):
        """The model parameters are profiled in the ELBO, no need to intialize them."""

    def _init_latent_parameters(self):
        self._latent_mean = torch.log(self._endog + (self._endog == 0)).to(DEVICE)
        self._latent_sqrt_variance = (
            1 / 2 * torch.ones((self.n_samples, self.dim)).to(DEVICE)
        )

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

    def _get_two_dim_covariances(self, sklearn_components):
        components_var = np.expand_dims(
            self.latent_sqrt_variance**2, 1
        ) * np.expand_dims(sklearn_components, 0)
        covariances = np.matmul(components_var, np.expand_dims(sklearn_components.T, 0))
        return covariances

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
        >>> print(pln.latent_positions.shape)
        >>> pln.viz(remove_exog_effect = True) # Visualize the latent positions
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
    def _additional_properties_list(self):
        return []

    @property
    def _additional_methods_list(self):
        return []

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pln.plot_correlation_circle(variables_names = ["MALAT1", "ACTB"])
        >>> pln.plot_correlation_circle(variables_names = ["A", "B"], indices_of_variables = [0,4])
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
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pln.biplot(variables_names = ["MALAT1", "ACTB"])
        >>> pln.biplot(variables_names = ["A", "B"], indices_of_variables = [0,4], colors = data["labels"])
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
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pln.pca_pairplot(n_components = 5)
        >>> pln.pca_pairplot(n_components = 5, colors = data["labels"])
        """,
    )
    def pca_pairplot(self, n_components: bool = 3, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        BaseModel,
        returns="""
        torch.Tensor
            The transformed endogenous variables (latent variables of the model).
        """,
        example="""
              >>> from pyPLNmodels import Pln, load_scrna
              >>> data = load_scrna()
              >>> pln = Pln.from_formula("endog ~ 1", data = data)
              >>> pln.fit()
              >>> transformed_endog = pln.transform()
              >>> print(transformed_endog.shape)
              >>> pln.viz()
              >>> transformed_no_exog = pln.transform(remove_exog_effect = True)
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
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln(data["endog"])
            >>> pln.fit()
            >>> pln.plot_expected_vs_true()
            >>> pln.plot_expected_vs_true(colors = data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1 + labels", data = data)
            >>> pln.fit()
            >>> pln.viz()
            >>> pln.viz(colors = data["labels"])
            >>> pln.viz(show_cov = True)
            >>> pln.viz(remove_exog_effect = True, colors = data["labels"])
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
