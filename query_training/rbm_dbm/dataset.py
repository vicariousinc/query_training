""" This file takes care of loading any required dataset, just call get_dataset with the appropriate dataset"""

import os

import numpy as np
from scipy.ndimage import convolve
from sklearn.datasets import load_digits as _load_digits
from sklearn.model_selection import train_test_split

from query_training import BASE

DATASET_PATH = os.path.join(BASE, 'datasets')


def get_dataset(filename):
    datasets = [
        'adult',
        'connect4',
        'digits',
        'dna',
        'mushrooms',
        'nips',
        'ocr_letters',
        'rcv1',
        'web',
    ]
    assert filename in datasets
    n_hidden = 200 if filename == 'nips' else 50
    if filename == 'web':
        n_hidden = 100
    if filename == 'digits':
        return uci_digits_eval() + (n_hidden,)

    filename += '.npz'
    mush_dict = np.load(os.path.join(DATASET_PATH, filename))
    X_train = mush_dict['train_data']
    X_valid = mush_dict['valid_data']
    X_test = mush_dict['test_data']
    xmax = np.max(np.concatenate([X_train, X_valid, X_test], axis=0), axis=0)
    xmin = np.min(np.concatenate([X_train, X_valid, X_test], axis=0), axis=0)
    X_train = (X_train - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_valid = (X_valid - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_test = (X_test - xmin) / (xmax + 0.0001)  # 0-1 scaling

    return X_train, X_valid, X_test, n_hidden


def nudge_dataset(X):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """

    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    return X


def uci_digits_eval():
    digits = _load_digits()
    X = np.asarray(digits.data, 'float32')

    fakeX = nudge_dataset(X)
    xmin, xmax = np.min(fakeX, 0), np.max(fakeX, 0)

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=0)
    X_train = nudge_dataset(X_train)
    X_train = (X_train - xmin) / (xmax + 0.0001)  # 0-1 scaling
    X_val = nudge_dataset(X_val)
    X_val = (X_val - xmin) / (xmax + 0.0001)  # 0-1 scaling

    return X_train, X_train.copy(), X_val
