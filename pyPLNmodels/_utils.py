import math
import textwrap

import time
import numpy as np

import torch


def _get_log_sum_of_endog(endog: torch.Tensor) -> torch.Tensor:
    """
    Compute offsets from the sum of endog.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)

    Returns
    -------
    torch.Tensor
        Offsets of size (n, p)
    """
    sum_of_endog = torch.sum(endog, axis=1)
    return torch.log(sum_of_endog.repeat((endog.shape[1], 1))).T


class _TimeRecorder:  # pylint: disable=too-few-public-methods
    def __init__(self, time_to_remove_from_beginning):
        self._running_times = []
        self._beginning_time = time.time() - time_to_remove_from_beginning

    def _has_been_launched(self):
        return len(self._running_times) > 0

    def _track_running_time(self):
        self._running_times.append(time.time() - self._beginning_time)


def _log_stirling(integer: torch.Tensor) -> torch.Tensor:
    """
    Compute log(n!) using the Stirling formula.

    Parameters
    ----------
    integer : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Approximation of log(n!) element-wise.
    """
    integer_ = integer + (integer == 0)  # Replace 0 with 1 since 0! = 1!
    return torch.log(torch.sqrt(2 * np.pi * integer_)) + integer_ * torch.log(
        integer_ / math.exp(1)
    )


def _add_doc(parent_class, *, params=None, example=None, returns=None, see_also=None):
    def wrapper(fun):
        doc = getattr(parent_class, fun.__name__).__doc__
        if doc is None:
            doc = ""
        doc = textwrap.dedent(doc).rstrip(" \n\r")
        if params is not None:
            doc += textwrap.dedent(params.rstrip(" \n\r"))
        if returns is not None:
            doc += "\n\nReturns"
            doc += "\n-------"
            doc += textwrap.dedent(returns)
        if see_also is not None:
            doc += "\n\nSee also"
            doc += "\n--------"
            doc += textwrap.dedent(see_also)
        if example is not None:
            doc += "\n\nExamples"
            doc += "\n--------"
            doc += textwrap.dedent(example)
        fun.__doc__ = doc
        return fun

    return wrapper


def _nice_string_of_dict(dictionnary: dict) -> str:
    """
    Create a nicely formatted string representation of a dictionary.

    Parameters
    ----------
    dictionnary : dict
        Dictionary to format.

    Returns
    -------
    str
        Nicely formatted string representation of the dictionary.
    """
    return_string = ""
    for each_row in zip(*([i] + [j] for i, j in dictionnary.items())):
        for element in list(each_row):
            return_string += f"{str(element):>12}"
        return_string += "\n"
    return return_string
