from __future__ import print_function

import os

import numpy as np

import torch
from query_training import BASE
from query_training.cmrf.cmrf8_bo import CMRF

BASE = os.path.join(BASE, 'cmrf')


def compute_iou(m, gt):
    return np.logical_and(gt, m).sum((1, 2)).astype(float) / np.logical_or(gt, m).sum(
        (1, 2)
    )


if __name__ == "__main__":
    b = np.load(os.path.join(BASE, "noisy_mnist_8_0.2.npz"))
    input_test = b["noisy_images_test"]
    output_test = b["images_test"]
    p_contour = b["p_contour"]

    n_clones = np.array([64, 1, 1])
    n_bp_iter = 15

    cmrf = CMRF(
        input_test, output_test, p_contour, n_clones, min_value_bu=0.0, device="cpu"
    )
    cmrf.load(os.path.join(BASE, "cmrf8_weights_15_mb50_lr1em2_nc64_emp.npz"))

    evidences = torch.from_numpy(cmrf.images_bu).to(cmrf.device).type(cmrf.dtype)

    damping = 1.0
    messages = cmrf.infer(evidences, n_bp_iter, damping, "sum", detach=True)
    beliefs = messages.prod(4) + 1e-20
    beliefs /= beliefs.sum(3, keepdim=True)
    beliefs = beliefs.detach().cpu().numpy()

    m = beliefs[:, :, :, -2] < 0.5
    m[:, 0, :] = 0
    m[:, -1, :] = 0
    m[:, :, 0] = 0
    m[:, :, -1] = 0

    gt = (output_test != 1).astype(int)
    ious = compute_iou(m, gt)

    targets = torch.from_numpy(cmrf.images_target).to(cmrf.device).type(cmrf.dtype)
    logp = (targets * beliefs).sum(3, keepdim=True).log().sum()
    nce = logp / (beliefs.shape[0] * beliefs.shape[1] * beliefs.shape[2])

    # Baseline
    baseline = np.random.binomial(1, 0.5, size=m.shape)
    ious_baseline = compute_iou(baseline, gt)

    print("Number of test images:", len(ious))
    print("IoU for QT is", ious.mean())
    print("NCE for QT is", -float(nce))
    print("IoU for baseline is", ious_baseline.mean())
    print("NCE for baseline is 1.00")  # by definition
