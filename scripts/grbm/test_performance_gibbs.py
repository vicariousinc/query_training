import os

import joblib
import numpy as np
from tqdm import tqdm

from query_training import BASE
from query_training.grbm.baseline import get_mean_nll_single_gaussian
from query_training.grbm.gibbs import get_mean_nll_gibbs, gibbs_sampling_with_mask

BASE = os.path.join(BASE, 'grbm')

BOX_TEST = False
N_RUNS = 1
n_warmup_samples = int(5e4)
n_samples = int(5e4)

dataset = joblib.load(os.path.join(BASE, 'frey_faces.joblib'))
test_data = dataset['test'][0]
if BOX_TEST:
    masks = 1 - joblib.load(os.path.join(BASE, 'frey_faces_box_test_masks.joblib'))
    results_folder = 'grbm_gibbs_box_test'
else:
    masks = joblib.load(os.path.join(BASE, 'frey_faces_test_masks.joblib'))
    results_folder = 'grbm_gibbs'

baseline_nlls = np.array(
    [get_mean_nll_single_gaussian(v, mask) for v, mask in tqdm(zip(test_data, masks))]
)
folder = os.path.join(BASE, 'results')
os.makedirs(folder, exist_ok=True)
joblib.dump(baseline_nlls, os.path.join(folder, 'frey_face_baseline_nlls.joblib'))


def process_image(v, mask, w, hid_b, vis_b):
    _, means_for_v = gibbs_sampling_with_mask(
        n_samples, v, mask, w, hid_b, vis_b, n_warmup_samples
    )
    return get_mean_nll_gibbs(v, mask, means_for_v)


for run in range(N_RUNS):
    for step in range(100000, 0, -100):
        checkpoint_fname = os.path.join(
            BASE,
            'training_checkpoints',
            str(run),
            'checkpoint_step_{}.joblib'.format(step),
        )
        checkpoint = joblib.load(checkpoint_fname)
        w = checkpoint['rbm']['rbm.w:0']
        hid_b = checkpoint['rbm']['rbm.hid_b:0']
        vis_b = checkpoint['rbm']['rbm.vis_b:0']
        # NLLs from Gibbs
        gibbs_nlls = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(process_image)(v, mask, w, hid_b, vis_b)
            for v, mask in tqdm(zip(test_data, masks))
        )
        gibbs_nlls = np.array(gibbs_nlls)
        folder = os.path.join(BASE, 'results', results_folder, str(run))
        os.makedirs(folder, exist_ok=True)
        joblib.dump(
            gibbs_nlls,
            os.path.join(
                folder, 'gibbs_nlls_for_{}'.format(os.path.basename(checkpoint_fname))
            ),
        )
