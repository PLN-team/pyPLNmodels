from typing import Union, Optional, Tuple, Dict
import warnings
import torch
import numpy as np
import pandas as pd
from patsy import dmatrices  # pylint: disable=no-name-in-module


from pyPLNmodels._utils import _get_log_sum_of_endog

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using GPU.")
else:
    DEVICE = "cpu"


def _handle_data(
    endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    exog: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series],
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    compute_offsets_method: str,
    add_const: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[pd.Index]]:
    """
    Handle the input data for the model.

    Parameters
    ----------
    endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The endog data. If a pandas.DataFrame is provided,
        the column names are stored for later use.
    exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
        The exog data.
    offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The offsets data.
    compute_offsets_method: str
        Method to compute offsets if not provided. Options are:
            - "zero" that will set the offsets to zero.
            - "logsum" that will take the logarithm of the sum (per line) of the counts.
    add_const : bool
        Indicates whether to add a constant column to the exog.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[pd.Index], Op]
        A tuple containing the processed endog, exog, offsets,
        and column names of endog and exog  (if available).

    Raises
    ------
    ValueError
        If the shapes of endog, exog, and offsets do not match or
        if any of `endog`, `exog`, and `offsets` are different from torch.Tensor,
        np.ndarray, or pd.DataFrame. `exog` may be a pd.Series without launching errors.
    """
    column_names_endog = endog.columns if isinstance(endog, pd.DataFrame) else None
    column_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else None

    endog, exog, offsets = _format_model_params(
        endog, exog, offsets, compute_offsets_method, add_const
    )
    _check_data_shapes(endog, exog, offsets)

    endog, offsets, column_names_endog = _remove_zero_columns(
        endog, offsets, column_names_endog
    )

    return endog, exog, offsets, column_names_endog, column_names_exog


def _format_model_params(
    endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    exog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    compute_offsets_method: str,
    add_const: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Format each of the model parameters to an array or None if None.

    Parameters
    ----------
    endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The endog data.
    exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
        The exog data.
    offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The offsets data.
    compute_offsets_method: str
        Method to compute offsets if not provided. Options are:
            - "zero" that will set the offsets to zero.
            - "logsum" that will take the logarithm of the sum (per line) of the counts.
    add_const : bool
        Indicates whether to add a constant column to the exog.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Formatted model parameters as torch.Tensor.

    Raises
    ------
    ValueError
        If endog has negative values or `compute_offsets_method` is none of
        None, "logsum" or "zero".
    """
    endog = _format_data(endog)
    if torch.min(endog) < 0:
        raise ValueError("Counts should be only non-negative values.")

    exog = _format_data(exog)
    if add_const:
        exog = _add_constant_to_exog(exog, endog.shape[0])

    if exog is not None:
        _check_full_rank_exog(exog)

    offsets = _compute_or_format_offsets(offsets, endog, compute_offsets_method)

    return endog, exog, offsets


def _check_data_shapes(
    endog: torch.Tensor, exog: torch.Tensor, offsets: torch.Tensor
) -> None:
    """
    Check if the shape of the input data is valid.

    Parameters
    ----------
    endog : torch.Tensor, shape (n, p)
        Count data.
    exog : torch.Tensor or None, shape (n, d) or None
        Covariate data.
    offsets : torch.Tensor or None, shape (n, p) or None
        Offset data.
    """
    n_endog, p_endog = endog.shape
    n_offsets, p_offsets = offsets.shape
    _check_dimensions_equal("endog", "offsets", n_endog, n_offsets, 0, 0)
    _check_dimensions_equal("endog", "offsets", p_endog, p_offsets, 1, 1)

    if exog is not None:
        n_cov, d_cov = exog.shape
        if n_cov < d_cov:
            raise ValueError(
                f"The number of samples ({n_cov}) should be greater "
                f"than the number of covariates ({d_cov})."
            )
        _check_dimensions_equal("endog", "exog", n_endog, n_cov, 0, 0)


def _format_data(
    data: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
) -> torch.Tensor:
    """
    Transforms the data into a torch.Tensor if the input is an array, and None if the input is None.
    Raises an error if the input is not an np.ndarray, torch.Tensor, pandas.DataFrame or None.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, torch.Tensor, pd.Series or None
        Input data.

    Returns
    -------
    torch.Tensor or None
        Formatted data.

    Raises
    ------
    AttributeError
        If the value is not an np.ndarray, torch.Tensor, pandas.DataFrame, pd.Series or None.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).to(DEVICE).float()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(DEVICE).float()
    if isinstance(data, torch.Tensor):
        return data.to(DEVICE).float()
    if isinstance(data, pd.Series):
        return torch.from_numpy(data.values).to(DEVICE).unsqueeze(1).float()
    raise AttributeError(
        "Please insert either a numpy.ndarray, pandas.DataFrame, pandas.Series or torch.Tensor"
    )


def _add_constant_to_exog(exog: torch.Tensor, length: int) -> torch.Tensor:
    ones = torch.ones(length, 1).to(DEVICE)
    if exog is None:
        return ones
    if length != exog.shape[0]:
        raise ValueError("The length of the exog should be the same as the length.")
    return torch.cat((exog, ones), dim=1)


def _check_full_rank_exog(exog: torch.Tensor, inflation: bool = False) -> None:
    mat = exog.T @ exog
    d = mat.shape[1]
    rank = torch.linalg.matrix_rank(mat)
    if rank != d:
        name_mat = "exog_inflation" if inflation else "exog"
        add_const_name = "add_const_inflation" if inflation else "add_const"
        msg = (
            f"Input matrix {name_mat} does not result in {name_mat}.T @{name_mat} being full rank "
            f"(rank = {rank}, expected = {d}). You may consider to remove one or more variables "
            f"or set {add_const_name} to False if that is not already the case. "
            f"You can also set 0 + {name_mat} in the formula to avoid adding an intercept."
        )
        raise ValueError(msg)


def _compute_or_format_offsets(
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    endog: torch.Tensor,
    compute_offsets_method: str,
) -> torch.Tensor:
    if offsets is None:
        if compute_offsets_method == "logsum":
            print("Setting the offsets as the log of the sum of endog.")
            return _get_log_sum_of_endog(endog)
        if compute_offsets_method == "zero":
            print("Setting the offsets to zero.")
            return torch.zeros(endog.shape).to(DEVICE)
        raise ValueError(
            f'Wrong method to compute offsets. Expected either "zero" or "logsum", '
            f"got {compute_offsets_method}."
        )

    return _format_data(offsets)


def _check_dimensions_equal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_order_first: int,
    dim_order_second: int,
) -> None:
    """
    Check if two dimensions are equal.

    Parameters
    ----------
    str_first_array : str
        Name of the first array.
    str_second_array : str
        Name of the second array.
    dim_first_array : int
        Dimension checked of the first array.
    dim_second_array : int
        Dimension checked of the second array.
    dim_order_first : int
        Dimension causing the error for the first tensor.
    dim_order_second : int
        Dimension causing the error for the second tensor.

    Raises
    ------
    ValueError
        If the dimensions of the two arrays are not equal.
    """
    if dim_first_array != dim_second_array:
        _raise_dimension_error(
            str_first_array,
            str_second_array,
            dim_first_array,
            dim_second_array,
            dim_order_first,
            dim_order_second,
        )


def _raise_dimension_error(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    str_first_array: str,
    str_second_array: str,
    dim_first_array: int,
    dim_second_array: int,
    dim_order_first: int,
    dim_order_second: int,
) -> None:
    """
    Raise an error for mismatched dimensions between two tensors.

    Parameters
    ----------
    str_first_array : str
        Name of the first tensor
    str_second_array : str
        Name of the second tensor
    dim_first_array : int
        Dimension of the first tensor
    dim_second_array : int
        Dimension of the second tensor
    dim_order_first : int
        Dimension causing the error for the first tensor.
    dim_order_second : int
        Dimension causing the error for the second tensor.

    Raises
    ------
    ValueError
        If the dimensions of the two tensors do not match.
    """
    msg = (
        f"The size of tensor {str_first_array} at non-singleton"
        f"dimension {dim_order_first} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_order_second}."
    )
    raise ValueError(msg)


def _remove_zero_columns(
    endog: torch.Tensor, offsets: torch.Tensor, column_names_endog: Optional[pd.Index]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[pd.Index]]:
    """
    Remove columns that contain only zeros from the endog and offsets tensors.

    Parameters
    ----------
    endog : torch.Tensor
        The endog data.
    offsets : torch.Tensor
        The offsets data.
    column_names_endog : Optional[pd.Index]
        Column names of the endog data if available.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[pd.Index]]
        The endog and offsets tensors with zero columns removed, and the updated column names.
    """
    zero_columns = torch.sum(endog, axis=0) == 0
    if torch.sum(zero_columns) > 0:
        dims = torch.arange(endog.shape[1]).to(DEVICE)[zero_columns]
        warnings.warn(
            f"The ({len(dims)}) following (index) variables contain "
            f"only zeros and are removed: {dims.cpu().numpy()}."
        )
        endog = endog[:, ~zero_columns]
        offsets = offsets[:, ~zero_columns]
        if column_names_endog is not None:
            column_names_endog = column_names_endog[~(zero_columns.cpu())]
        print(f"Now dataset of size {endog.shape}.")
    return endog, offsets, column_names_endog


def _extract_data_from_formula(
    formula: str,
    data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
) -> Tuple:
    """
    Extract data from the given formula and data dictionary.

    Parameters
    ----------
    formula : str
        The formula specifying the data to extract.
    data : Dict[str, Any]
        A dictionary containing the data.

    Returns
    -------
    Tuple
        A tuple containing the extracted endog, exog, and offsets.

    """
    variables = dmatrices(formula, data=data)
    endog = variables[0]
    exog = variables[1]
    non_zero_exog = (exog**2).sum(axis=0) > 0
    exog = exog[:, non_zero_exog]

    if exog.size == 0:
        exog = None
    if "offsets" in data.keys():
        offsets = data["offsets"]
        print("Taking the offsets from the data given.")
    else:
        offsets = None
    return endog, exog, offsets
