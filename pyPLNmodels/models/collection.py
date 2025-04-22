from typing import Dict, Any, Union, Iterable, Optional, List
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import pandas as pd
import numpy as np

from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.utils._data_handler import _handle_data, _extract_data_from_formula
from pyPLNmodels.utils._viz import _show_collection
from pyPLNmodels.utils._utils import _add_doc, _nice_string_of_dict


class Collection(ABC):
    """
    Abstract class that stores a collection
    of models inheriting from the BaseModel class.
    For now, a collection is implemented for
        - PlnPCA
        - PlnMixture
        - PlnNetwork
    This allows to tests a grid of hyperparameter.
    The better hyperparameter is given by minimizing the
    BIC criterion.

    See also
    --------
    :class:`~pyPLNmodels.PlnPCACollection`
    :class:`~pyPLNmodels.ZIPlnPCACollection`
    :class:`~pyPLNmodels.PlnNetworkCollection`
    :class:`~pyPLNmodels.PlnMixtureCollection`
    """

    _type_grid = type

    _grid_value_name = str

    PlnModel: BaseModel

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        grid: Iterable[float],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes the collection.

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
            Overridden (useless) if `offsets` is not None.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the `exog`. Defaults to `True`.
        """
        self._dict_models = {}
        (
            self._endog,
            self._exog,
            self._offsets,
            self.column_names_endog,
            self.column_names_exog,
        ) = _handle_data(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        self._fitted = False
        self._init_models(grid)

    def _init_models(
        self,
        grid: Iterable[float],
    ):
        """
        Method for initializing the models.

        Parameters
        ----------
        grid : Iterable[float]
            The range of grid to fit.
        """
        if isinstance(grid, Iterable):
            for grid_value in grid:
                if self._is_right_instance(grid_value):
                    self._dict_models[grid_value] = self._instantiate_model(grid_value)
                    self._set_column_names(self._dict_models[grid_value])
                else:
                    raise TypeError(
                        f"Please instantiate `{self._grid_value_name}` with a list "
                        f"of {self._type_grid}."
                    )
            self._dict_models = OrderedDict(sorted(self._dict_models.items()))
        else:
            raise TypeError(
                f"Please instantiate the `{self._grid_value_name}` with an iterable"
                f" (such as a list of {self._type_grid})."
            )

    def _is_right_instance(self, grid_value):
        return isinstance(grid_value, self._type_grid)

    def _set_column_names(self, model):
        model.column_names_endog = self.column_names_endog
        model.column_names_exog = self.column_names_exog

    @classmethod
    @_add_doc(
        BaseModel,
        params="""
              grid : Iterable[float], optional(keyword-only)
                The hyperparameter that needs to be tested.
              """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        grid: Iterable[float],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable=missing-function-docstring
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog=endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            grid=grid,
            add_const=False,
        )

    @abstractmethod
    def _instantiate_model(self, grid_value):
        pass

    @property
    @_add_doc(BaseModel)
    def exog(self) -> torch.Tensor:  # pylint:disable = missing-function-docstring
        return self[self.grid[0]].exog

    @property
    @_add_doc(BaseModel)
    def offsets(self) -> torch.Tensor:
        """
        Property representing the `offsets`.

        Returns
        -------
        torch.Tensor
            The `offsets`.
        """
        return self[self.grid[0]].offsets

    @property
    @_add_doc(BaseModel)
    def endog(self) -> torch.Tensor:
        """
        Property representing the `endog`.

        Returns
        -------
        torch.Tensor
            The `endog`.
        """
        return self[self.grid[0]].endog

    @property
    def n_samples(self):
        """Number of samples in the dataset."""
        return self[self.grid[0]].n_samples

    @property
    def grid(self) -> List[float]:
        """
        Property representing the grid
        given in initialization.

        Returns
        -------
        List[float]
            The grid.
        """
        return list(self._dict_models.keys())

    @property
    def coef(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the coefficients of the collection.

        Returns
        -------
        Dict[float, torch.Tensor]
            The coefficients.
        """
        return {key: value.coef for key, value in self.items()}

    def values(self):
        """
        Models in the collection as a list.

        Returns
        -------
        ValuesView
            The models in the collection.
        """
        return self._dict_models.values()

    def items(self):
        """
        Get the key-value pairs of the models in the collection.

        Returns
        -------
        ItemsView
            The key-value pairs of the models.
        """
        return self._dict_models.items()

    def __getitem__(self, grid_value: int) -> Any:
        """
        Model with the specified grid_value.

        Parameters
        ----------
        grid_value : int
            The value of the grid wanted.

        Returns
        -------
        BaseModel
            The model with the specified grid value.
        """
        return self._dict_models[grid_value]

    def __len__(self) -> int:
        """
        Number of models in the collection.

        Returns
        -------
        int
            The number of models in the collection.
        """
        return len(self._dict_models)

    def __iter__(self):
        """
        Iterate over the models in the collection.

        Returns
        -------
        Iterator
            Iterator over the models.
        """
        return iter(self._dict_models)

    def __contains__(self, grid_value: int) -> bool:
        """
        Check if a model with the specified grid_value exists in the collection.

        Parameters
        ----------
        grid_value : float
            The grid_value to check.

        Returns
        -------
        bool
            `True` if a model with the specified grid_value exists, `False` otherwise.
        """
        return grid_value in self.keys()

    def keys(self):
        """
        Get the grid of the collection.

        Returns
        -------
        KeysView
            The grid of the collection.
        """
        return self._dict_models.keys()

    def get(self, key: Any, default: Any) -> Any:
        """
        Get the model with the specified key, or return a default value if the key does not exist.

        Parameters
        ----------
        key : Any
            The key to search for.
        default : Any
            The default value to return if the key does not exist.

        Returns
        -------
        Any
            The model with the specified key, or the default value if the key does not exist.
        """
        if key in self:
            return self[key]
        return default

    def _print_beginning_message(self) -> str:
        print(f"Adjusting {len(self)} {self.PlnModel.__name__} models.\n")

    @property
    @_add_doc(BaseModel)
    def dim(self) -> int:  # pylint:disable = missing-function-docstring
        return self[self.grid[0]].dim

    @property
    @_add_doc(BaseModel)
    def nb_cov(self) -> int:  # pylint:disable = missing-function-docstring
        return self[self.grid[0]].nb_cov

    @abstractmethod
    def _init_next_model_with_current_model(
        self, next_model: BaseModel, current_model: BaseModel
    ):
        """
        Initialize the next `PlnModel` model with the parameters of the current `PlnModel` model.

        Parameters
        ----------
        next_model : PlnModel
            The next model to initialize.
        current_model : PlnModel
            The current model.
        """

    def fit(
        self,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        """
        Fit each model in the collection.

        Parameters
        ----------
        maxiter : int, optional
            The maximum number of iterations to be done, by default 400.
        lr : float, optional(keyword-only)
            The learning rate, by default 0.01.
        tol : float, optional(keyword-only)
            The tolerance, by default 1e-6.
        verbose : bool, optional(keyword-only)
            Whether to print verbose output, by default `False`.

        Returns
        -------
        Collection
        """
        self._print_beginning_message()
        for i in range(len(self.values())):
            model = self[self.grid[i]]
            model.fit(
                maxiter=maxiter,
                lr=lr,
                tol=tol,
                verbose=verbose,
            )
            if i < len(self.values()) - 1:
                next_model = self[self.grid[i + 1]]
                self._init_next_model_with_current_model(next_model, model)

        self._print_ending_message()

        return self

    def _print_ending_message(self):
        delimiter = "=" * 70
        bic = self.BIC
        aic = self.AIC
        print(
            f"{delimiter}\n\nDONE!\n"
            f"Best model (lower BIC): {self._grid_value_name} "
            f"{list(bic.keys())[np.argmin(list(bic.values()))]}\n"
            f"    Best model(lower AIC): {self._grid_value_name} "
            f" {list(aic.keys())[np.argmin(list(aic.values()))]}\n\n"
            f"{delimiter}\n"
        )

    @property
    def BIC(self) -> Dict[int, int]:
        """
        Property representing the BIC scores of the models in the collection.

        Returns
        -------
        Dict[int, float]
            The BIC scores of the models.
        """
        return {grid_value: int(self[grid_value].BIC) for grid_value in self.grid}

    @property
    def ICL(self) -> Dict[int, int]:
        """
        Property representing the ICL scores of the models in the collection.

        Returns
        -------
        Dict[int, float]
            The ICL scores of the models.
        """
        return {grid_value: int(self[grid_value].ICL) for grid_value in self.grid}

    @property
    def AIC(self) -> Dict[int, int]:
        """
        Property representing the AIC scores of the models in the collection.

        Returns
        -------
        Dict[int, float]
            The AIC scores of the models.
        """
        return {grid_value: int(self[grid_value].AIC) for grid_value in self.grid}

    def best_model(self, criterion: str = "BIC") -> Any:
        """
        Get the best model according to the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use ('AIC' or 'BIC'), by default 'BIC'.

        Returns
        -------
        Any
            The best model.
        """
        return self[self._best_grid_value(criterion)]

    def _best_grid_value(self, criterion):
        if criterion not in ("BIC", "AIC", "ICL"):
            raise ValueError(f"Unknown criterion {criterion}")
        if criterion == "BIC":
            criterion = self.BIC
        if criterion == "AIC":
            criterion = self.AIC
        if criterion == "ICL":
            criterion = self.ICL
        return self.grid[np.argmin(list(criterion.values()))]

    @property
    def loglike(self) -> Dict[int, float]:
        """
        Property representing the log-likelihoods of the models in the collection.

        Returns
        -------
        Dict[int, float]
            The log-likelihoods of the models.
        """
        return {grid_value: self[grid_value].loglike for grid_value in self.grid}

    def show(self, figsize: tuple = (10, 10)):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.

        Parameters
        ----------
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,10)
        """
        _show_collection(self, figsize=figsize, absc_label=self._grid_value_name)

    @property
    def _useful_methods_strings(self) -> str:
        return [".show()", ".best_model()", ".keys()", ".items()", ".values()"]

    @property
    def _useful_attributes_string(self) -> str:
        return [".BIC", ".AIC", ".loglike"]

    def __repr__(self) -> str:
        """
        Return a string representation of the `Collection` object.

        Returns
        -------
        str
            The string representation of the `Collection` object.
        """
        nb_models = len(self)
        delimiter = "\n" + "-" * 70 + "\n"

        # Header
        to_print = (
            f"{delimiter}"
            f"Collection of {nb_models} {self.PlnModel.__name__} models with {self.dim} variables."
            f"{delimiter}"
        )

        to_print += f" - {self._grid_value_name} considered: {self.grid}\n"

        dict_bic = {self._grid_value_name: "criterion"} | self.BIC
        grid_value_bic = self._best_grid_value("BIC")
        to_print += " - BIC criterion:      "
        to_print += (
            f"\n{_nice_string_of_dict(dict_bic, best_grid_value=grid_value_bic)}\n"
        )
        to_print += f"Best model (lower BIC): {grid_value_bic}\n\n"

        # AIC metric
        dict_aic = {self._grid_value_name: "criterion"} | self.AIC
        grid_value_aic = self._best_grid_value("AIC")
        to_print += " - AIC criterion:      "
        to_print += (
            f"\n{_nice_string_of_dict(dict_aic, best_grid_value=grid_value_aic)}\n"
        )
        to_print += f"   Best model (lower AIC): {grid_value_aic}"

        # Footer
        to_print += (
            f"{delimiter}"
            f"* Useful attributes\n"
            f"    {self._useful_attributes_string}\n"
            f"* Useful methods\n"
            f"    {self._useful_methods_strings}"
            f"{delimiter}"
        )

        return to_print

    @property
    def _name(self):
        return str(type(self).__name__)
