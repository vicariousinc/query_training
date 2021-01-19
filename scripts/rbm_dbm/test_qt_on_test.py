from __future__ import print_function

import glob
import os

import joblib
import numpy as np

import torch
from query_training import BASE
from query_training.rbm_dbm.dataset import get_dataset
from query_training.rbm_dbm.rbm import RBM

BASE = os.path.join(BASE, 'dbm')
N_RUNS = 1

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


def test_given_model(dataset, Whv, b, masks, n_bp_iter, strength=1, model='rbm'):
    _, _, x_test, n_hidden = get_dataset(dataset)
    if model == 'dbm':
        x_hidden_test = 0.5 * np.ones((x_test.shape[0], n_hidden))
        x_test = np.concatenate((x_test, x_hidden_test), axis=1)
        masks = np.concatenate((masks, np.zeros_like(x_hidden_test)), axis=1)
    else:
        assert model == 'rbm', 'Unsupported model {}'.format(model)

    rbm = RBM(x_test, H=n_hidden)
    rbm.Whv[:] = torch.from_numpy(Whv)
    rbm.b[:] = torch.from_numpy(b)
    return rbm.test(strength=strength, n_bp_iter=n_bp_iter, masks=masks)


n_bp_iter = 50

for dataset in datasets:
    results = {}
    _, _, x_test, n_hidden = get_dataset(dataset)
    masks = np.random.binomial(1, 0.5 * np.ones_like(x_test))

    # Experiments for QT
    for checkpoint_fname in glob.glob(
        os.path.join(BASE, 'qt_checkpoints', dataset, 'rbm_*.npz')
    ):
        checkpoint = np.load(checkpoint_fname)
        Whv = checkpoint['Whv']
        b = checkpoint['b']
        strength = checkpoint['strength']
        results[checkpoint_fname] = test_given_model(
            dataset, Whv, b, masks, n_bp_iter=n_bp_iter, strength=strength, model='dbm'
        )
        print('{}, {}'.format(checkpoint_fname, results[checkpoint_fname]))
        joblib.dump(
            results,
            os.path.join(
                BASE, 'results', 'qt_results_for_{}_on_test.joblib'.format(dataset)
            ),
        )

    for run in range(N_RUNS):
        results = {}
        # Experiments for advil
        results_folder = glob.glob(
            os.path.join(BASE, 'training_checkpoints', str(run), '*{}*'.format(dataset))
        )[0]
        results_on_val = joblib.load(
            os.path.join(
                BASE,
                'results',
                'advil_qt_results_for_{}_run_{}.joblib'.format(dataset, run),
            )
        )
        min_nce = 10000
        min_key = None
        for key in results_on_val:
            if key.endswith('npz'):
                continue

            if results_on_val[key] < min_nce:
                min_nce = results_on_val[key]
                min_key = key

        print('Min NCE on val: {}'.format(min_nce))
        checkpoint_fname = min_key
        checkpoint = joblib.load(checkpoint_fname)
        hid_b1 = checkpoint['dbm']['dbm.hid_b1:0']
        hid_b2 = checkpoint['dbm']['dbm.hid_b2:0']
        vis_b = checkpoint['dbm']['dbm.vis_b:0']
        w1 = checkpoint['dbm']['dbm.w1:0']
        w2 = checkpoint['dbm']['dbm.w2:0']
        W = np.concatenate((w1.T, w2), axis=1)
        Whv = 0.5 * W
        c_v = np.concatenate((vis_b, hid_b2))
        c_h = hid_b1
        c_v = c_v + Whv.sum(0)
        c_h = c_h + Whv.sum(1)
        b = np.concatenate((c_v, c_h)).reshape((-1, 1))
        results[checkpoint_fname] = test_given_model(
            dataset, Whv, b, masks, n_bp_iter=n_bp_iter, strength=1, model='dbm'
        )
        joblib.dump(
            results,
            os.path.join(
                BASE,
                'results',
                'advil_qt_results_for_{}_run_{}_on_test.joblib'.format(dataset, run),
            ),
        )
        print('{}, {}'.format(checkpoint_fname, results[checkpoint_fname]))
