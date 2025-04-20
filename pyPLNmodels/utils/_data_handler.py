from typing import Union, Optional, Tuple, Dict
import warnings
import torch
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix, PatsyError  # pylint: disable=no-name-in-module
from sklearn.preprocessing import LabelEncoder

from pyPLNmodels.utils._utils import _get_log_sum_of_endog

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using GPU.")
else:
    DEVICE = "cpu"


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _handle_data(
    endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    exog: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series],
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    compute_offsets_method: str,
    add_const: bool,
    remove_zero_columns: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[pd.Index]]:
    """
    Handle the input data for the model.

    Parameters
    ----------
    endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The `endog` data. If a `pandas.DataFrame` is provided,
        the column names are stored for later use.
    exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
        The `exog` data.
    offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The `offsets` data.
    compute_offsets_method: str
        Method to compute offsets if not provided. Options are:
            - "zero" that will set the offsets to zero.
            - "logsum" that will take the logarithm of the sum (per line) of the counts.
    add_const : bool
        Indicates whether to add a constant column to the `exog`.
    remove_zero_columns: bool
        Whether to remove or not the columns which have only zeros.
        Default to `True`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[pd.Index]]
        A tuple containing the processed `endog`, `exog`, `offsets`,
        and column names of `endog` and `exog` (if available).

    Raises
    ------
    ValueError
        If the shapes of `endog`, `exog`, and `offsets` do not match or
        if any of `endog`, `exog`, and `offsets` are different from `torch.Tensor`,
        `np.ndarray`, or `pd.DataFrame`. `exog` may be a `pd.Series` without launching errors.
    """

    column_names_endog = endog.columns if isinstance(endog, pd.DataFrame) else None
    column_names_exog = exog.columns if isinstance(exog, pd.DataFrame) else None

    endog, exog, offsets = _format_model_params(
        endog, exog, offsets, compute_offsets_method, add_const
    )
    if column_names_endog is None:
        column_names_endog = [f"Dim_{i+1}" for i in range(endog.shape[1])]
    if column_names_exog is None:
        nb_cov = 0 if exog is None else exog.shape[1]
        column_names_exog = [f"Exog_{i+1}" for i in range(nb_cov)]
        if add_const is True:
            column_names_exog[0] = "Intercept"

    _check_data_shapes(endog, exog, offsets)
    if remove_zero_columns is True:
        endog, offsets, column_names_endog = _remove_zero_columns(
            endog, offsets, column_names_endog
        )

    if exog is not None:
        exog = _remove_useless_exog(exog, column_names_exog, is_inflation=False)
        _check_full_rank_exog(exog, name_mat="exog")
    if torch.max(offsets) > 10:
        warnings.warn(
            "Offsets are very large. Consider taking the logarithm. NaN may appear."
        )

    return endog, exog, offsets, column_names_endog, column_names_exog


def _handle_inflation_data(exog_inflation, add_const_inflation, endog):
    """
    Format only the zero inflation part. Raises a `ValueError`
    if there is no zero inflation and no `add_const_inflation`.
    """
    column_names_exog_inflation = (
        exog_inflation.columns if isinstance(exog_inflation, pd.DataFrame) else None
    )
    exog_inflation = _format_data(exog_inflation)
    if add_const_inflation is True:
        exog_inflation = _add_constant_to_exog(exog_inflation, endog.shape[0])
    else:
        if exog_inflation is None:
            raise ValueError("Please fit a PLN model if there is no inflation.")
    if column_names_exog_inflation is None:
        column_names_exog_inflation = [
            f"Exog_infl_{i+1}" for i in range(exog_inflation.shape[1])
        ]
    if add_const_inflation is True:
        column_names_exog_inflation[0] = "Intercept"
    _check_dimensions_equal(
        "endog", "exog_inflation", endog.shape[0], exog_inflation.shape[0], 0, 0
    )
    exog_inflation = _remove_useless_exog(
        exog_inflation, column_names_exog_inflation, is_inflation=True
    )
    _check_full_rank_exog(
        exog_inflation, name_mat="exog_inflation", add_const_name="add_const_inflation"
    )
    dirac = endog == 0
    return exog_inflation, column_names_exog_inflation, dirac


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
        The `endog` data.
    exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]
        The `exog` data.
    offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
        The `offsets` data.
    compute_offsets_method: str
        Method to compute offsets if not provided. Options are:
            - "zero" that will set the offsets to zero.
            - "logsum" that will take the logarithm of the sum (per line) of the counts.
    add_const : bool
        Indicates whether to add a constant column to the `exog`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Formatted model parameters as `torch.Tensor`.

    Raises
    ------
    ValueError
        If `endog` has negative values or `compute_offsets_method` is none of
        None, "logsum" or "zero".
    """
    endog = _format_data(endog)
    if endog.shape[0] == 0:
        raise ValueError("`endog` is empty.")
    if torch.sum(endog < 0) > 0:
        raise ValueError("Counts should be only non-negative values.")

    exog = _format_data(exog)
    if add_const is True:
        exog = _add_constant_to_exog(exog, endog.shape[0])

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
    data: Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series],
) -> torch.Tensor:
    """
    Transforms the data into a `torch.Tensor` if the input is an array, and None if the input is
    None. Raises an error if the input is not an `np.ndarray`, `torch.Tensor`, `pandas.DataFrame`
    or None.

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
        If the value is not an `np.ndarray`, `torch.Tensor`, `pandas.DataFrame`,
        `pd.Series` or `None`.
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
        return _series_to_tensor(data).unsqueeze(1).to(DEVICE).float()
        # return torch.from_numpy(data.values).to(DEVICE).unsqueeze(1).float()
    msg = "Please insert either a `numpy.ndarray`, `pandas.DataFrame`, `pandas.Series` "
    msg += "or `torch.Tensor`"
    raise AttributeError(msg)


def _series_to_tensor_and_encoder(series: pd.Series) -> torch.Tensor:
    if series.dtype in [int, np.float64]:
        return torch.tensor(series.values).to(DEVICE), None
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(series)
    return torch.tensor(integer_encoded).to(DEVICE), label_encoder


def _series_to_tensor(series: pd.Series) -> torch.Tensor:
    out, _ = _series_to_tensor_and_encoder(series)
    return out


def _add_constant_to_exog(exog: torch.Tensor, length: int) -> torch.Tensor:
    ones = torch.ones(length, 1).to(DEVICE)
    if exog is None:
        return ones
    if length != exog.shape[0]:
        msg = f"The length of the `exog` ({exog.shape[0]}) "
        msg += f"should be the same as the length of `endog` ({length})."
        raise ValueError(msg)
    return torch.cat((ones, exog), dim=1)


def _check_full_rank_exog(
    exog: torch.Tensor, name_mat: str = "exog", add_const_name: str = "add_const"
) -> None:
    mat = exog.T @ exog
    d = mat.shape[1]
    rank = torch.linalg.matrix_rank(mat)
    if rank != d:
        if name_mat == "(exog,clusters)":
            formula = "exog | clusters "
        else:
            formula = name_mat
        msg = (
            f"Input matrix {name_mat} does not result in {name_mat}.T @{name_mat} being full rank "
            f"(rank = {rank}, expected = {d}). You may consider removing one or more variables "
            f"or set {add_const_name} to False if that is not already the case. "
            f"You can also set 0 + {formula} in the formula to avoid adding an intercept."
        )
        raise ValueError(msg)


def _check_full_rank_exog_and_ones(
    exog_and_ones: torch.Tensor,
) -> None:
    d = exog_and_ones.shape[1]
    mat = exog_and_ones.T @ exog_and_ones
    rank = torch.linalg.matrix_rank(mat)
    if rank != d:
        msg = (
            f"Input matrix exog does not result in (exog,1).T @(exog,1) being full rank "
            f"(rank = {rank}, expected = {d}). This gives non identifiable cluster means."
            f" You may consider removing one or more variables in exog."
        )
        raise ValueError(msg)


def _compute_or_format_offsets(
    offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    endog: torch.Tensor,
    compute_offsets_method: str,
) -> torch.Tensor:
    if offsets is None:
        if compute_offsets_method == "logsum":
            print("Setting the offsets as the log of the sum of `endog`.")
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
        Name of the first tensor.
    str_second_array : str
        Name of the second tensor.
    dim_first_array : int
        Dimension of the first tensor.
    dim_second_array : int
        Dimension of the second tensor.
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
        f" dimension {dim_order_first} ({dim_first_array}) must match "
        f"the size of tensor {str_second_array} ({dim_second_array}) at "
        f"non-singleton dimension {dim_order_second}."
    )
    raise ValueError(msg)


def _remove_zero_columns(
    endog: torch.Tensor, offsets: torch.Tensor, column_names_endog: Optional[pd.Index]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[pd.Index]]:
    """
    Remove columns that contain only zeros from the `endog` and `offsets` tensors.

    Parameters
    ----------
    endog : torch.Tensor
        The `endog` data.
    offsets : torch.Tensor
        The `offsets` data.
    column_names_endog : Optional[pd.Index]
        Column names of the `endog` data if available.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[pd.Index]]
        The `endog` and `offsets` tensors with zero columns removed, and the updated column names.
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
        A tuple containing the extracted `endog`, `exog`, and `offsets`.
    """
    endog_name = formula.split("~")[0].strip()
    if isinstance(data[endog_name], pd.DataFrame):
        column_names_endog = data[endog_name].columns
    else:
        column_names_endog = None

    variables = dmatrices(formula, data=data)
    endog, exog = variables[0], variables[1]
    column_names_exog = exog.design_info.column_names

    if exog.size == 0:
        exog = None
    else:
        exog = pd.DataFrame(exog)
        exog.columns = column_names_exog
    if "offsets" in data.keys():
        offsets = data["offsets"]
        print("Taking the offsets from the data given.")
    else:
        offsets = None
    if column_names_endog is not None:
        endog = pd.DataFrame(endog)
        endog.columns = column_names_endog
    return endog, exog, offsets


def _extract_right_term_formula(
    right_term_formula: str,
    data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
    exog_string: str,
) -> Tuple:
    """
    Extract only the `exog_inflation` from the given formula and data dictionary.

    Parameters
    ----------
    right_term_formula : str
        The formula specifying the data to extract.
    data : Dict[str, Any]
        A dictionary containing the data.
    exog_string: str
        The name of the exog that is being extracted.
        Should be either 'exog_inflation' or 'clusters'.

    Returns
    -------
    torch.Tensor
        `exog_inflation` of size (n_samples, nb_cov_inflation).
    """
    try:
        exog = dmatrix(right_term_formula, data=data)
        column_names_exog = exog.design_info.column_names
        exog = pd.DataFrame(exog)
        exog.columns = column_names_exog
        return exog
    except PatsyError as err:
        msg = f"Formula of `{exog_string}` did not work: {right_term_formula}."
        msg += " Error from Patsy:"
        warnings.warn(msg)
        raise err


def _array2tensor(func):
    def setter(self, array_like=None):
        array_like = _format_data(array_like)
        return func(self, array_like)

    return setter


def _remove_useless_exog(exog, column_names_exog, is_inflation):
    zero_columns = (torch.sum(exog**2, axis=0) == 0).cpu()
    if torch.sum(zero_columns) > 0:
        for i, is_zero in enumerate(zero_columns):
            if is_zero:
                msg = f"removing column {column_names_exog[i]} as it is only zeros"
                if is_inflation is True:
                    msg += " for the inflation part."
                print(msg)
        return exog[:, ~zero_columns]
    return exog


def _extract_data_inflation_from_formula(formula, data):
    if "|" not in formula:
        msg = "`exog_inflation` and `exog` are set to the same array. "
        msg += "If you need different `exog_inflation`, "
        msg += (
            "specify it with a pipe: '|' like in the following: endog ~ 1 + x | x + y "
        )
        print(msg)
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        exog_inflation = exog
    else:
        split_formula = formula.split("|")
        formula_exog = split_formula[0]
        endog, exog, offsets = _extract_data_from_formula(formula_exog, data)
        formula_infla = split_formula[1]
        exog_inflation = _extract_right_term_formula(
            formula_infla, data, "exog_inflation"
        )

    return endog, exog, offsets, exog_inflation


def _extract_data_and_clusters_from_formula(formula, data):
    if "|" not in formula:
        msg = "Clusters should be specified in the formula, like "
        msg += "in the following example: 'endog ~ 1 + exog | clusters'"
        raise ValueError(msg)
    split_formula = formula.split("|")
    formula_exog = split_formula[0]
    endog, exog, offsets = _extract_data_from_formula(formula_exog, data)
    formula_clusters = split_formula[1]
    if _validate_after_pipe(formula_clusters) is False:
        msg = "You may consider only one word in the second part of the formula. "
        msg += f"Got {formula_clusters}. "
        raise ValueError(msg)
    formula_clusters = "0 + " + formula_clusters  # no intercept. as it is clusters.
    clusters = _extract_right_term_formula(formula_clusters, data, "clusters")

    return endog, exog, offsets, clusters


def _format_clusters_and_encoder(clusters):
    if isinstance(clusters, pd.Series):
        return _series_to_tensor_and_encoder(clusters)
    clusters = _format_data(clusters).squeeze()
    if clusters.dim() == 2 and torch.all((clusters == 0) | (clusters == 1)):
        return clusters, None  # Already one-hot encoded

    # Convert to one-hot encoding
    if clusters.dim() == 1:
        clusters = clusters.to(torch.int64)
        num_classes = clusters.max().item() + 1
        one_hot_clusters = torch.nn.functional.one_hot(
            clusters, num_classes=num_classes
        )
        return one_hot_clusters.float(), None

    msg = "Input clusters format is not recognized. Give either"
    msg += " a one dimensional tensor or a one-hot encoded tensor."
    raise ValueError(msg)


def _validate_after_pipe(part):
    if len(part.split()) == 1:
        return True
    return False


def _check_dimensions_for_prediction(endog, model_endog, exog, model_exog):
    _check_dimensions_equal(
        "new endog", "model endog", endog.shape[1], model_endog.shape[1], 1, 1
    )
    if exog is None and model_exog is not None:
        raise ValueError(
            "`exog` is `None` but exogenous variables were given in the model."
        )
    if model_exog is None and exog is not None:
        raise ValueError(
            "`exog` is not `None` but exogenous variables were not given in the model."
        )
    if exog is not None and model_exog is not None:
        _check_dimensions_equal(
            "new exog", "model exog", exog.shape[1], model_exog.shape[1], 1, 1
        )


def _get_dummies(input_tensor):
    """
    Convert a tensor of class indices to a one-hot encoded tensor.

    Parameters:
    input_tensor (torch.Tensor): Tensor containing class indices.

    Returns:
    torch.Tensor: One-hot encoded tensor.
    """
    # Determine the number of classes
    num_classes = input_tensor.max().item() + 1

    # Initialize the output tensor with zeros
    one_hot_tensor = torch.zeros(
        input_tensor.size(0), num_classes, device=input_tensor.device
    )

    # Fill the appropriate positions with ones
    one_hot_tensor.scatter_(1, input_tensor.unsqueeze(1), 1)

    return one_hot_tensor


def _check_int(rank):
    if rank < 1:
        raise AttributeError(f"The rank should be an int >= 1. Got {rank}")
    if isinstance(rank, int) is False:
        raise AttributeError(f"The rank should be an int. Got {rank}")
