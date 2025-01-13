from typing import Optional, Union
import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel
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
    >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
    >>> pln = Pln(endog,add_const = True)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors = labels)

    >>> from pyPLNmodels import Pln, get_simulation_parameters, sample_pln
    >>> param = get_simulation_parameters()
    >>> endog = sample_pln(param)
    >>> data = {"endog": endog}
    >>> pln = Pln.from_formula("endog ~ 1", data)
    >>> pln.fit()
    >>> print(pln)
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
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_variance,
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
        """,
    )
    def latent_variables(self):
        return self.latent_mean

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        return self.dim * (self.dim + 2 * self.nb_cov + 1) / 2

    @property
    def _additional_properties_string(self):
        return ""

    @property
    def _additional_methods_string(self):
        return ""
