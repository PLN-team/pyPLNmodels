import itertools

import torch


def _get_formula(nb_cov, add_const):
    if nb_cov == 0:
        if add_const is True:
            return "endog ~ 1"
        return "endog ~ 0"
    if add_const is False:
        return "endog ~ 0 + exog"
    return "endog ~ exog"


def _get_formula_inflation(nb_cov, add_const):
    first_formula = _get_formula(nb_cov, add_const)
    second_formula = "| 0 + exog_inflation"
    return first_formula + second_formula


def _get_formula_lda(nb_cov):
    first_formula = _get_formula(nb_cov, False)
    second_formula = "| clusters"
    return first_formula + second_formula


def median(t):
    """Mean squared error of a torch.Tensor."""
    return torch.median(torch.abs(t))


def _generate_combinations(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()

    combinations = list(itertools.product(*values))

    result = [dict(zip(keys, combination)) for combination in combinations]
    return result


def _get_formula_from_kw(kwargs, is_inflated, is_lda):
    nb_cov = kwargs.get("nb_cov")
    add_const = kwargs.get("add_const")
    if is_inflated is True:
        formula = _get_formula_inflation(nb_cov, add_const)
    elif is_lda is True:
        formula = _get_formula_lda(nb_cov)
    else:
        formula = _get_formula(nb_cov, add_const)
    return formula


def _get_argmax_mapping(tensor):
    # Calculate the sum of each row
    row_sums = tensor.sum(dim=1)
    # Get the indices that would sort the row sums
    sorted_indices = torch.argsort(row_sums)
    # Create the argmax mapping
    argmax_mapping = torch.argsort(sorted_indices)
    return argmax_mapping
