import glob
import os

import joblib
import numpy as np
from tqdm import tqdm

from query_training import BASE
from query_training.rbm_dbm.dataset import get_dataset
from query_training.rbm_dbm.dbm_gibbs import (
    get_mean_nll_gibbs,
    gibbs_sampling_with_mask,
)

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
]

step_ind = {
    'adult': 20000,
    'connect4': 15000,
    'digits': 62500,
    'dna': 11000,
    'mushrooms': 39000,
    'nips': 10520,
    'ocr_letters': 72000,
    'rcv1': 36000,
}

n_samples = int(1e4)
n_warmup_samples = int(1e4)


def process_sample(v, mask, w1, w2, hid_b1, hid_b2, vis_b):
    means_for_v = gibbs_sampling_with_mask(
        n_samples,
        v,
        mask,
        w1,
        w2,
        hid_b1,
        hid_b2,
        vis_b,
        n_warmup_samples=n_warmup_samples,
        random_h1=True,
    )
    prob, nll = get_mean_nll_gibbs(v, mask, means_for_v)
    return prob, nll


for dataset in datasets:
    _, _, x_test, n_hidden = get_dataset(dataset)
    masks = np.random.binomial(1, 0.5 * np.ones_like(x_test))
    for run in range(N_RUNS):
        results_folder = glob.glob(
            os.path.join(BASE, 'training_checkpoints', str(run), '*{}*'.format(dataset))
        )[0]
        # Experiments for advil
        best_ind = step_ind[dataset]
        checkpoint_fname = os.path.join(
            results_folder, 'checkpoint_step_{}.joblib'.format(best_ind)
        )
        checkpoint = joblib.load(checkpoint_fname)
        hid_b1 = checkpoint['dbm']['dbm.hid_b1:0']
        hid_b2 = checkpoint['dbm']['dbm.hid_b2:0']
        vis_b = checkpoint['dbm']['dbm.vis_b:0']
        w1 = checkpoint['dbm']['dbm.w1:0']
        w2 = checkpoint['dbm']['dbm.w2:0']
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(process_sample)(
                v=v, mask=mask, w1=w1, w2=w2, hid_b1=hid_b1, hid_b2=hid_b2, vis_b=vis_b
            )
            for v, mask in tqdm(zip(x_test, masks))
        )
        prob, nll = list(zip(*output))
        prob = np.array(prob)
        nll = np.array(nll)
        print('advil, {}, nll: {}'.format(dataset, np.mean(nll)))
        folder = os.path.join(BASE, 'results', str(run))
        os.makedirs(folder, exist_ok=True)
        joblib.dump(
            {
                'w1': w1,
                'w2': w2,
                'hid_b1': hid_b1,
                'hid_b2': hid_b2,
                'vis_b': vis_b,
                'prob': prob,
                'nll': nll,
            },
            os.path.join(folder, 'advil_gibbs_{}_run_{}.joblib'.format(dataset, run)),
        )
