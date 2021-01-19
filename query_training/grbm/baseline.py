import numpy as np


def get_mean_nll_single_gaussian(v, mask):
    mu = np.mean(v[mask == 0])
    sigma = np.std(v[mask == 0])
    return np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2)
        + 0.5 * (v[mask == 1] - mu) ** 2 / sigma ** 2
    )
