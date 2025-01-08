from abc import ABC
from typing import Union, Optional


import torch
import numpy as np
import pandas as pd

from pyPLNmodels._data_handler import _handle_data, _extract_data_from_formula
from pyPLNmodels._criterion import _LossCriterionMonitor


class BaseModel(ABC):
    """
    Abstract base class for all the PLN based models that will be derived.
    """

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes the model class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to `None`.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to `None`.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overriden (useless) if `offsets` is not None.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the `exog`. Defaults to `True`.
        """

        (
            self._endog,
            self._exog,
            self._offsets,
            self._column_names_endog,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            compute_offsets_method,
            add_const,
        )

        self._criterion_monitor = _LossCriterionMonitor()
        self._fitted = False

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):
        """
        Create a model instance from a formula and data.

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary. Each value can be either a torch.Tensor,
            a np.ndarray or pd.DataFrame
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overriden (useless) if data["offsets"] is not None.
        """
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    def fit(self, nb_epoch: int):
        """
        Fits the model.
        """
        for _ in range(nb_epoch):
            print("x")

    def _init_parameters(self):
        pass

    def viz(self):
        """
        Visualizes the latent variables.
        """
        print("Visualizing...")
