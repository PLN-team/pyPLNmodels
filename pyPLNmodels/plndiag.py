from typing import Union, Optional

import pandas as pd
import torch
import numpy as np

from pyPLNmodels.pln import Pln
from pyPLNmodels.base import BaseModel
from pyPLNmodels._closed_forms import _closed_formula_diag_covariance
from pyPLNmodels.elbos import profiled_elbo_pln_diag
from pyPLNmodels._utils import _add_doc


class PlnDiag(Pln):
    """
    PLN model with diagonal covariance. Inference is faster, as well as memory requirements.

    Examples
    --------
    >>> from pyPLNmodels import PlnDiag, load_scrna
    >>> data = load_scrna()
    >>> pln = PlnDiag(data["endog"])
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])

    >>> from pyPLNmodels import PlnDiag, load_scrna
    >>> data = load_scrna()
    >>> pln = PlnDiag.from_formula("endog ~ 1 + labels", data=data)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors=data["labels"])
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

    @property
    def _covariance(self):
        return _closed_formula_diag_covariance(
            marginal_mean=self._marginal_mean,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            n_samples=self.n_samples,
        )

    def compute_elbo(self):
        return profiled_elbo_pln_diag(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
        )
