import torch
import functools


def MSE(t):
    return torch.mean(t**2)


def filter_models(models_name):
    def decorator(my_test):
        @functools.wraps(my_test)
        def new_test(**kwargs):
            fixture = next(iter(kwargs.values()))
            if type(fixture).__name__ not in models_name:
                return None
            return my_test(**kwargs)

        return new_test

    return decorator
