import torch


def MSE(t):
    return torch.mean(t**2)
