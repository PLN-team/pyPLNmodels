from typing import Union, Iterable, Optional, Dict

import torch
import numpy as np
import pandas as pd

from pyPLNmodels.models.collection import Collection
from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.models.plnnetwork import PlnNetwork
from pyPLNmodels.models.plnpcacollection import PlnPCACollection
from pyPLNmodels.utils._utils import _add_doc
from pyPLNmodels.utils._data_handler import _extract_data_from_formula
from pyPLNmodels.utils._viz import _show_collection_and_nb_links


class PlnNetworkCollection(Collection):
    """
    A collection of `PlnNetwork` models, each with a different penalty.
    For more details, see:
    J. Chiquet, S. Robin, M. Mariadassou: "Variational Inference for sparse network
    reconstruction from count data"

    Examples
    --------
    >>> from pyPLNmodels import PlnNetworkCollection, load_scrna
    >>> data = load_scrna()
    >>> nets = PlnNetworkCollection.from_formula("endog ~ 1", data = data, penalties = [1,10, 100])
    >>> nets.fit()
    >>> print(nets)
    >>> nets.show()
    >>> print(nets.best_model())
    >>> print(nets[10])

    See also
    --------
    :class:`~pyPLNmodels.PlnNetwork`
    :func:`pyPLNmodels.PlnNetworkCollection.from_formula`
    :func:`pyPLNmodels.PlnNetworkCollection.__init__`
    :class:`~pyPLNmodels.PlnPCACollection`
    :class:`~pyPLNmodels.ZIPlnPCACollection`
    :class:`~pyPLNmodels.PlnMixtureCollection`
    """

    _type_grid = float
    _grid_value_name = "penalty"
    PlnModel = PlnNetwork

    @_add_doc(
        Collection,
        params="""
            penalties : Iterable[float], optional(keyword-only)
                The range of penalties, by default `(1, 10, 100, 1000)`.
            """,
        example="""
            >>> from pyPLNmodels import PlnNetworkCollection, load_scrna
            >>> data = load_scrna()
            >>> nets = PlnNetworkCollection(endog = data["endog"], penalties = [1,10,100])
            >>> nets.fit()
            >>> print(nets.best_model())
        """,
        returns="""
            PlnNetworkCollection
        """,
        see_also="""
        :func:`pyPLNmodels.PlnNetworkCollection.from_formula`
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
        penalties: Optional[Iterable[float]] = (1, 10, 100, 1000),
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog=endog,
            grid=penalties,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
              penalties : Iterable[float], optional(keyword-only)
                The penalties that needs to be tested. By default
                (1, 10, 100, 1000).
              """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        penalties: Optional[Iterable[int]] = (1, 10, 100, 1000),
    ):  # pylint: disable=missing-function-docstring, arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            penalties=penalties,
            add_const=False,
        )

    def _instantiate_model(self, grid_value):
        return PlnNetwork(
            endog=self._endog,
            exog=self._exog,
            offsets=self._offsets,
            penalty=grid_value,
            add_const=False,
        )

    def _is_right_instance(self, grid_value):
        return isinstance(grid_value, (int, float))

    def _init_next_model_with_current_model(
        self, next_model: PlnNetwork, current_model: PlnNetwork
    ):
        # coef is a closed form
        next_model.components_prec = torch.clone(current_model.components_prec)
        next_model.latent_mean = torch.clone(current_model.latent_mean)
        next_model.latent_sqrt_variance = torch.clone(
            current_model.latent_sqrt_variance
        )

    @property
    def precision(self) -> Dict[float, torch.Tensor]:
        """
        Property representing the precision of each model in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The precision for each model.
        """
        return {key: value.precision for key, value in self.items()}

    @property
    def penalties(self):
        """
        Property representing the penalties of each model in the collection.

        Returns
        -------
        List[float]
            The penalties.
        """
        return self.grid

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import PlnNetworkCollection, load_scrna
            >>> data = load_scrna()
            >>> nets = PlnNetworkCollection(endog = data["endog"], penalties = [1,10,100])
            >>> nets.fit()
        """,
        returns="""
        PlnNetworkCollection
        """,
    )
    def fit(
        self,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @property
    @_add_doc(PlnPCACollection)
    def latent_mean(
        self,
    ) -> Dict[int, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        return {model.penalty: model.latent_mean for model in self.values()}

    @property
    @_add_doc(PlnPCACollection)
    def latent_variance(
        self,
    ) -> Dict[int, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        return {model.penalty: model.latent_variance for model in self.values()}

    @_add_doc(
        Collection,
        example="""
            >>> from pyPLNmodels import PlnNetworkCollection, load_scrna
            >>> data = load_scrna()
            >>> nets = PlnNetworkCollection(endog = data["endog"], penalties = [1, 10, 100])
            >>> nets.fit()
            >>> print(nets.best_model())
        """,
        returns="""
        PlnNetworkCollection
        """,
    )
    def best_model(self, criterion: str = "BIC") -> PlnNetwork:
        return super().best_model(criterion=criterion)

    @property
    def components_prec(self) -> Dict[float, torch.Tensor]:
        """
        Property representing the components of the precision matrix for each model
        in the collection.

        Returns
        -------
        Dict[int, torch.Tensor]
            The components of the precision.
        """
        return {key: value.components_prec for key, value in self.items()}

    @property
    def nb_links(self):
        """Number of links of each model."""
        return {model.penalty: model.nb_links for model in self.values()}

    def show(self, figsize: tuple = (15, 10)):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.
        Also show the number of links in the model.
        The AIC and BIC criteria might not always provide meaningful guidance
        for selecting the penalty. Instead, we recommend focusing on the desired number of links.

        Parameters
        ----------
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,15)
        """
        absc_label = "Penalties"
        _show_collection_and_nb_links(self, figsize=figsize, absc_label=absc_label)
