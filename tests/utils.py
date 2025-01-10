import functools

import torch


def filter_models(models_name):
    """
    Decorator to filter test functions based on the type of the first argument.

    Args:
        models_name (list): A list of model names to filter the tests.

    Returns:
        function: A decorator that wraps the test function and filters it based on the model name.

    Example:
        @filter_models(['ModelA', 'ModelB'])
        def test_function(model):
            # test code here
    """

    def decorator(my_test):
        @functools.wraps(my_test)
        def new_test(**kwargs):
            fixture = next(iter(kwargs.values()))
            if type(fixture).__name__ not in models_name:
                return None
            return my_test(**kwargs)

        return new_test

    return decorator


def _get_formula(nb_cov):
    if nb_cov == 0:
        return "endog ~ 0"
    return "endog ~ 0 + exog"


def mse(t):
    """Mean squared error of a torch.Tensor."""
    return torch.mean(t**2)
