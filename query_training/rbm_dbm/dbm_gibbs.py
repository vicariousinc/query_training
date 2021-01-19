import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True, cache=True)
def hid1_to_vis(h1, w1, vis_b):
    return sigmoid(np.dot(w1, h1.astype(np.float32)) + vis_b)


@numba.jit(nopython=True, cache=True)
def hid1_to_hid2(h1, w2, hid_b2):
    return sigmoid(np.dot(h1.astype(np.float32), w2) + hid_b2)


@numba.jit(nopython=True, cache=True)
def vis_hid2_to_hid1(v, h2, w1, w2, hid_b1):
    return sigmoid(
        np.dot(w2, h2.astype(np.float32)) + np.dot(v.astype(np.float32), w1) + hid_b1
    )


@numba.jit(nopython=True, cache=True)
def gibbs_sampling_with_mask(
    n_samples,
    v,
    mask,
    w1,
    w2,
    hid_b1,
    hid_b2,
    vis_b,
    n_warmup_samples=5000,
    random_h1=False,
):
    # Initialize h1
    if random_h1:
        h1 = np.array([np.random.binomial(1, 0.5) for _ in range(hid_b1.shape[0])])
    else:
        h1 = gibbs_initialize_h1_from_vh2_with_mask(
            v, mask, w1, w2, hid_b1, hid_b2, vis_b
        )
    for ii in range(n_warmup_samples):
        h1 = gibbs_h1_vh2_h1_with_mask(h1, v, mask, w1, w2, hid_b1, hid_b2, vis_b)

    means_for_v = np.zeros((n_samples, v.shape[0]))
    for ii in range(n_samples):
        h1 = gibbs_h1_vh2_h1_with_mask(h1, v, mask, w1, w2, hid_b1, hid_b2, vis_b)
        means_for_v[ii] = hid1_to_vis(h1, w1, vis_b)

    return means_for_v


@numba.jit(nopython=True, cache=True)
def gibbs_h1_vh2_h1_with_mask(h1, v_true, mask, w1, w2, hid_b1, hid_b2, vis_b):
    # h1 to v, h2
    v = np.array([np.random.binomial(1, p) for p in hid1_to_vis(h1, w1, vis_b)])
    v[mask == 0] = v_true[mask == 0]
    h2 = np.array([np.random.binomial(1, p) for p in hid1_to_hid2(h1, w2, hid_b2)])
    # v, h2 to h1
    h1 = np.array(
        [np.random.binomial(1, p) for p in vis_hid2_to_hid1(v, h2, w1, w2, hid_b1)]
    )
    return h1


@numba.jit(nopython=True, cache=True)
def gibbs_initialize_h1_from_vh2_with_mask(v_true, mask, w1, w2, hid_b1, hid_b2, vis_b):
    v = np.array([np.random.binomial(1, 0.5) for _ in range(vis_b.shape[0])])
    v[mask == 0] = v_true[mask == 0]
    h2 = np.array([np.random.binomial(1, 0.5) for _ in range(hid_b2.shape[0])])
    # v, h2 to h1
    h1 = np.array(
        [np.random.binomial(1, p) for p in vis_hid2_to_hid1(v, h2, w1, w2, hid_b1)]
    )
    return h1


def get_mean_nll_gibbs(v, mask, means_for_v):
    v = v[mask == 1]
    means_for_v = means_for_v[:, mask == 1]
    prob = np.mean(means_for_v, axis=0)
    nll = -np.mean(v * np.log(prob + 1e-20) + (1 - v) * np.log(1 - prob + 1e-20))
    return prob, nll
