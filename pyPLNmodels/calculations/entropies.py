import math

import torch

from pyPLNmodels.utils._utils import _trunc_log


def entropy_bernoulli(latent_prob):
    """
    Entropy for a bernoulli distribution.
    """
    return torch.sum(
        -latent_prob * _trunc_log(latent_prob)
        - (1 - latent_prob) * _trunc_log(1 - latent_prob)
    )


def entropy_gaussian(latent_variance):
    """
    Entropy for a gaussian variables.
    """
    product_dimensions = torch.prod(torch.tensor(list(latent_variance.shape)))
    return (
        1 / 2 * torch.sum(latent_variance)
        + product_dimensions / 2 * math.log(2 * math.pi)
        + product_dimensions / 2
    )


def entropy_gaussian_mixture(latent_variance, latent_prob):
    """
    Entropy for a gaussian variable when there are mixture weights (PlnMixture).
    """
    product_dimensions = torch.prod(torch.tensor(list(latent_variance.shape[-2:])))
    return (
        1 / 2 * torch.sum((latent_prob.T).unsqueeze(2) * latent_variance)
        + product_dimensions / 2 * math.log(2 * math.pi)
        + product_dimensions / 2
    )


def entropy_mixture_gaussian(latent_prob, latent_variance, covariance):
    """
    Entropy for a mixture of two gaussians, one with variance latent_variance
    and the other with diagonal covariance.
    """
    first_mixture = 1 / 2 * torch.sum((1 - latent_prob) * torch.log(latent_variance))
    second_mixture = -1 / 2 * torch.sum(latent_prob, axis=0) @ torch.diag(covariance)
    return first_mixture + second_mixture


def entropy_clusters(latent_prob, weights):
    """
    Entropy for the weights of a gaussian mixture.
    """
    entropy = torch.sum(torch.xlogy(latent_prob, weights.unsqueeze(0)))
    entropy -= torch.sum(torch.xlogy(latent_prob, latent_prob))
    return entropy
