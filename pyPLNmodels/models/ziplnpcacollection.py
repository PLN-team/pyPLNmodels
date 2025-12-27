from typing import Union, Iterable, Optional, Dict

import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.collection import Collection
from pyPLNmodels.models.base import DEFAULT_TOL, BaseModel
from pyPLNmodels.models.ziplnpca import ZIPlnPCA
from pyPLNmodels.models.plnpcacollection import PlnPCACollection
from pyPLNmodels.utils._utils import _add_doc, _init_next_model_pca
from pyPLNmodels.utils._data_handler import (
    _handle_inflation_data,
    _format_data,
    _extract_data_inflation_from_formula,
)


class ZIPlnPCACollection(
    PlnPCACollection
):  # pylint: disable=too-many-instance-attributes
    """
    A collection of `ZIPlnPCA` models, each with a different number of components.
    The number of components can also be referred to as the number of PCs, or the
    rank of the covariance matrix.
    For more details, see B. Bricout ?

    Let :math:`q` be an integer lower than the number of variables :math:`p`. A single model assumes the following:

    .. math::

        \begin{align}
        Z_i &\sim \mathcal{N}(X_i^{\top} B, \Sigma), \operatorname{rank}(\Sigma)=q\\
        W_{ij} &\sim \mathcal{B}(\sigma(X_i^{0^{\top}} B^0_j)), \\
        Y_{ij} \mid Z_{ij} &\sim (1- W_{ij})\mathcal{P}(\exp(o_{ij} + Z_{ij})).
        \end{align}

    The model parameters are:

    - :math:`B \in \mathbb{R}^{d \times p}` :code:`coef`: matrix of regression coefficients
    - :math:`B^0 \in \mathbb{R}^{d_0 \times p}` :code:`coef_inflation`: matrix of regression coefficients for the inflation part
    - :math:`\Sigma  \in \mathcal{S}_{+}^{p}` :code:`covariance`: covariance matrix of the latent variables :math:`Z_i`

    Data provided is

    - :math:`Y \in \mathbb{R}^{n \times p}` :code:`endog`: matrix of endogenous variables (counts). Required.
    - :math:`X \in \mathbb{R}^{n \times d}` :code:`exog`: matrix of exogenous variables (covariates). Defaults to vector of 1's.
    - :math:`X^0 \in \mathbb{R}^{n \times d_0}` :code:`exog_inflation`: matrix of exogenous variables (covariates) for the inflation part. Defaults to vector of 1's.
    - :math:`O  \in \mathbb{R}^{n \times p}` :code:`offsets`: offsets (in log space). Defaults to matrix of 0's.

    The number of covariates is denoted by :math:`d` (:code:`nb_cov`) and the number
    of covariates for the inflation is denoted by :math:`d_0`(:code:`nb_cov_infla`),
    while :math:`n` denotes the number of samples (:code:`n_samples`)
    and :math:`p` denotes the number of dimensions (:code:`dim`), i.e. features or number of variables.

    The :code:`ZIPlnPCACollection` implements several :code:`ZIPlnPCA` models with different ranks :math:`q`.

    Examples
    --------
    >>> from pyPLNmodels import ZIPlnPCACollection, load_scrna
    >>> data = load_scrna()
    >>> zipcas = ZIPlnPCACollection.from_formula("endog ~ 1", data = data, ranks = [5,8, 12])
    >>> zipcas.fit()
    >>> print(zipcas)
    >>> zipcas.show()
    >>> print(zipcas.best_model())
    >>> print(zipcas[5])

    See also
    --------
    :class:`~pyPLNmodels.ZIPlnPCA`
    :func:`pyPLNmodels.ZIPlnPCACollection.from_formula`
    :func:`pyPLNmodels.ZIPlnPCACollection.__init__`
    :class:`~pyPLNmodels.PlnNetworkCollection`
    :class:`~pyPLNmodels.PlnPCACollection`
    :class:`~pyPLNmodels.PlnMixtureCollection`
    """

    _type_grid = int
    _grid_value_name = "rank"
    PlnModel = ZIPlnPCA

    @_add_doc(
        Collection,
        params="""
            ranks : Iterable[int], optional(keyword-only)
                The range of ranks, by default `(3, 5)`.
            """,
        example="""
            >>> from pyPLNmodels import ZIPlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> zipcas = ZIPlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> zipcas.fit()
            >>> print(zipcas.best_model())
        """,
        returns="""
            ZIPlnPCACollection
        """,
        see_also="""
        :func:`pyPLNmodels.ZIPlnPCACollection.from_formula`
        """,
    )
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
        ranks: Optional[Iterable[int]] = (3, 5),
        use_closed_form_prob: bool = True,
    ):  # pylint: disable=too-many-arguments
        self._use_closed_form_prob = use_closed_form_prob
        self._exog_inflation, self.column_names_exog_inflation, self._dirac = (
            _handle_inflation_data(
                exog_inflation, add_const_inflation, _format_data(endog)
            )
        )
        super().__init__(
            endog=endog,
            ranks=ranks,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
              ranks : Iterable[int], optional(keyword-only)
                The ranks (or number of PCs) that needs to be tested.
                By default (3, 5)
              """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        ranks: Optional[Iterable[int]] = (3, 5),
        use_closed_form_prob: bool = True,
    ):  # pylint: disable=missing-function-docstring, arguments-differ, too-many-arguments
        endog, exog, offsets, exog_inflation = _extract_data_inflation_from_formula(
            formula, data
        )
        return cls(
            endog=endog,
            exog=exog,
            exog_inflation=exog_inflation,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            ranks=ranks,
            add_const=False,
            add_const_inflation=False,
            use_closed_form_prob=use_closed_form_prob,
        )

    def _instantiate_model(self, grid_value):
        return ZIPlnPCA(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            exog_inflation=self._exog_inflation,
            rank=grid_value,
            add_const=False,
            add_const_inflation=False,
            use_closed_form_prob=self._use_closed_form_prob,
        )

    def _set_column_names(self, model):
        super()._set_column_names(model)
        model.column_names_exog_inflation = self.column_names_exog_inflation

    def _init_next_model_with_current_model(
        self, next_model: ZIPlnPCA, current_model: ZIPlnPCA
    ):
        next_model = _init_next_model_pca(next_model, current_model)
        next_model.coef_inflation = torch.clone(current_model.coef_inflation)
        if self._use_closed_form_prob is False:
            next_model.latent_prob = torch.clone(current_model.latent_prob)

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import ZIPlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> zipcas = ZIPlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> zipcas.fit()
        """,
        returns="""
        ZIPlnPCACollection
        """,
    )
    def fit(
        self,
        maxiter: int = 1000,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL / 1000,
        verbose: bool = False,
    ):
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    def coef_inflation(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the coef_inflation, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The coef inflation for each model.
        """
        return {model.rank: model.coef_inflation for model in self.values()}

    @property
    def latent_prob(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the `latent_prob`, for each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The coef inflation for each model.
        """
        return {model.rank: model.latent_prob for model in self.values()}

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import ZIPlnPCACollection, load_scrna
            >>> data = load_scrna()
            >>> zipcas = ZIPlnPCACollection(endog = data["endog"], ranks = [4,6,8])
            >>> zipcas.fit()
            >>> print(zipcas.best_model())
        """,
        returns="""
        ZIPlnPCACollection
        """,
    )
    def best_model(self, criterion: str = "BIC") -> ZIPlnPCA:
        return super().best_model(criterion=criterion)

    @property
    def nb_cov_inflation(self) -> int:  # pylint:disable = missing-function-docstring
        """
        The number of exogenous variables for the inflation part.
        """
        return self[self.grid[0]].nb_cov_inflation
