import numpy as np


def cos_schedule(dl, param_min, param_max):
    """ Given a Dataloader, return a per-batch cos param schedule"""
    cos_base = (1 + np.cos(np.linspace(3.14, 3.14 * 3, len(dl))))*.5
    return cos_base * (param_max - param_min) + param_min


def linear_schedule(dl, param_min, param_max):
    """ Given a Dataloader, return a per-batch linear param schedule"""
    return np.linspace(param_min, param_max, len(dl))


def neg_linear_schedule(dl, param_min, param_max):
    """ Given a Dataloader, return a per-batch neg-linear learning param schedule"""
    return np.linspace(param_max, param_min, len(dl))
