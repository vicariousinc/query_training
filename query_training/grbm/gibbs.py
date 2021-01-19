import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True, cache=True)
def vis2hid_potential(v, w, hid_b):
    return -np.dot(v.astype(np.float32), w) - hid_b


@numba.jit(nopython=True, cache=True)
def vis2hid(v, w, hid_b):
    return sigmoid(-vis2hid_potential(v, w, hid_b))


@numba.jit(nopython=True, cache=True)
def hid2vis(h, w, vis_b):
    return np.dot(h.astype(np.float32), w.T) + vis_b


@numba.jit(nopython=True, cache=True)
def gibbs_sampling_with_mask(
    n_samples, v, mask, w, hid_b, vis_b, n_warmup_samples=100000
):
    # Initialize h
    h = np.array(
        [
            np.random.binomial(1, p)
            for p in vis2hid(np.random.randn(vis_b.shape[0]) + vis_b, w, hid_b)
        ]
    )
    for ii in range(n_warmup_samples):
        h = gibbs_hvh_with_mask(h, v, mask, w, hid_b, vis_b)

    samples = np.zeros((n_samples, h.shape[0]))
    means_for_v = np.zeros((n_samples, v.shape[0]))
    for ii in range(n_samples):
        h = gibbs_hvh_with_mask(h, v, mask, w, hid_b, vis_b)
        samples[ii] = h
        means_for_v[ii] = hid2vis(h, w, vis_b)

    return samples, means_for_v


@numba.jit(nopython=True, cache=True)
def gibbs_hvh_with_mask(h, v_true, mask, w, hid_b, vis_b):
    v = gibbs_hv(h, w, hid_b, vis_b)
    v[mask == 0] = v_true[mask == 0]
    h = gibbs_vh(v, w, hid_b, vis_b)
    return h


@numba.jit(nopython=True, cache=True)
def gibbs_vh(v, w, hid_b, vis_b):
    h = np.array([np.random.binomial(1, p) for p in vis2hid(v, w, hid_b)])
    return h


@numba.jit(nopython=True, cache=True)
def gibbs_hv(h, w, hid_b, vis_b):
    mu = hid2vis(h, w, vis_b)
    v = np.random.randn(mu.shape[0]) + mu
    return v


def get_mean_nll_gibbs(v, mask, means_for_v):
    v = v[mask == 1]
    means_for_v = means_for_v[:, mask == 1]
    mixture_prob = np.exp(-0.5 * (v.reshape((1, -1)) - means_for_v) ** 2) / np.sqrt(
        2 * np.pi
    )
    return -np.mean(np.log(np.mean(mixture_prob, axis=0)))
