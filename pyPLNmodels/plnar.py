from typing import Optional, Union

import torch
import pandas as pd
import numpy as np

from pyPLNmodels.base import BaseModel
from pyPLNmodels.plndiag import PlnDiag
from pyPLNmodels.elbos import elbo_plnar
from pyPLNmodels._initialization import _init_coef
from pyPLNmodels._utils import _add_doc


class PlnAR(PlnDiag):
    """
    Autoregressive Pln model.
    """

    _autoreg_diff_term: torch.Tensor
    __coef: torch.Tensor
    _sqrt_covariance: torch.Tensor

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnAR, load_scrna
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
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
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
    def _coef(self):
        return self.__coef

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
