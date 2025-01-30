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


def mse(t):
    """Mean squared error of a torch.Tensor."""
    return torch.mean(t**2)


def _generate_combinations(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()

    combinations = list(itertools.product(*values))

    result = [dict(zip(keys, combination)) for combination in combinations]
    return result


def _get_formula_from_kw(kwargs, is_inflated):
    nb_cov = kwargs.get("nb_cov")
    add_const = kwargs.get("add_const")
    if is_inflated is True:
        formula = _get_formula_inflation(nb_cov, add_const)
    else:
        formula = _get_formula(nb_cov, add_const)
    return formula
