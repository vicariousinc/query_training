from __future__ import print_function

import os

import numpy as np

from query_training import BASE
from query_training.cmrf.cmrf8_bo import CMRF

BASE = os.path.joib(BASE, 'cmrf')
b = np.load(os.path.join(BASE, "noisy_mnist_8_0.2.npz"))

input_train = np.vstack((b["noisy_images_train"], b["noisy_images_val"]))
output_train = np.vstack((b["images_train"], b["images_val"]))
p_contour = b["p_contour"]

n_clones = np.array([64, 1, 1])
mb_size = 50

cmrf = CMRF(input_train, output_train, p_contour, n_clones, min_value_bu=0.0)

n_bp_iter = 15
cmrf.train(
    learning_rate=1e-2,
    n_epochs=10,
    n_bp_iter=n_bp_iter,
    damping=1.0,
    mb_size=mb_size,
    savetofile="cmrf8_weights_{}_mb{}_lr1em2_nc{}.npz".format(
        n_bp_iter, mb_size, n_clones[0]
    ),
    save_every=100,
)
