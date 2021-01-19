# Training a BM using stochastic query optimization. Torch version.

from __future__ import print_function

from builtins import range

import numpy as np
from scipy.special import logit as invsigmoid
from tqdm import tqdm

import torch
from torch.nn.functional import softplus

torch.autograd.set_detect_anomaly(True)


class RBM(object):
    def __init__(self, X, H, dtype=torch.float32, device=None):
        assert (0 <= X).all() and (X <= 1).all()
        assert H >= 0
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        N, self.V = X.shape
        self.H = H  # dimension of hidden
        self.X = X
        self.Whv = torch.randn(
            self.H, self.V, device=self.device, dtype=self.dtype, requires_grad=True
        )
        self.Whv.data *= 0.01
        self.b = torch.zeros(
            self.V + self.H, 1, device=self.device, dtype=self.dtype, requires_grad=True
        )
        self.strength = torch.ones(
            1, device=self.device, dtype=self.dtype, requires_grad=True
        )
        self.strength.data *= 5.0

    def train(
        self,
        x_val=None,
        learning_rate=5e-3,
        n_epochs=50,
        n_bp_iter=10,
        damping=1.0,
        beta=1.0,
        mb_size=500,
        p_target=0.5,
        savetofile=None,
        save_every=100,
    ):
        n_images = self.X.shape[0]
        optimizer = torch.optim.Adam(
            (self.Whv, self.b, self.strength), lr=learning_rate
        )
        # optimizer = torch.optim.SGD((self.Whv, self.b), lr=learning_rate)
        # optimizer = torch.optim.RMSprop((self.Whv, self.b), lr=learning_rate)
        convergence, convergence_val, best_rbm, best_score = [], [], None, np.inf
        assert n_epochs > save_every
        for epoch in range(n_epochs):
            order = np.random.permutation(n_images)
            for start in range(0, n_images, mb_size):
                optimizer.zero_grad()
                stop = min(start + mb_size, n_images)
                x1 = self.X[order[start:stop]]
                loss_mb = 0.0
                pr = x1.shape[0]
                if (
                    self.H == 200
                ):  # for nips dataset only, this parameter doesn't change anything, just makes it fit in the GPU
                    pr = 200
                assert x1.shape[0] % pr == 0
                for sample in range(int(x1.shape[0] / pr)):
                    x_mb = x1[sample * pr : (sample + 1) * pr]
                    loss = self.ncll(
                        self.strength, x_mb, n_bp_iter, p_target, damping, beta
                    )
                    loss_mb += loss.detach().cpu().numpy().item()
                    loss.backward()
                convergence.append(loss_mb / x_mb.shape[0])
                print(
                    "strength",
                    self.strength[0].item(),
                    "beta",
                    beta,
                    "mb",
                    mb_size,
                    "Wshape",
                    self.Whv.shape,
                    "It:",
                    len(convergence),
                    ",",
                    "Ep:",
                    epoch,
                    ",",
                    start,
                    "/",
                    n_images,
                    convergence[-1],
                )
                optimizer.step()
            if x_val is not None and (epoch + 1) % save_every == 0:
                convergence_val.append(
                    self.test(self.strength[0].item(), n_bp_iter, p_target, x_val)
                )
                if convergence_val[-1] < best_score:
                    best_score = convergence_val[-1]
                    best_rbm = (
                        self.Whv.detach().cpu().numpy().copy(),
                        self.b.detach().cpu().numpy(),
                        self.strength[0].item(),
                    )
                if savetofile is not None:
                    np.savez_compressed(
                        '{}_epoch_{}.npz'.format(savetofile, epoch),
                        Whv=self.Whv.detach().cpu().numpy(),
                        strength=self.strength[0].item(),
                        b=self.b.detach().cpu().numpy(),
                        H=self.H,
                        convergence=np.array(convergence),
                    )
        return convergence, convergence_val, best_rbm, best_score

    def infer(self, u, n_bp_iter, damping, beta, mode="sum", messages=None):
        N = u.shape[0]
        if messages is None:
            messages = torch.zeros(
                (2, N, self.H, self.V), device=self.device, dtype=self.dtype
            )
        for i in range(n_bp_iter):
            messages = layer(messages, self.Whv, self.b, u, mode, beta)
        return messages

    def ncll(self, strength, x, n_bp_iter, p_target, damping, beta, mode="sum"):
        u = invsigmoid(x).clip(-1000, 1000)
        evidence_mask = np.random.binomial(
            1, 1.0 - p_target, u.shape
        )  # 0: target node, 1: evidence node,
        u_masked = np.hstack(
            (u * evidence_mask, np.zeros((u.shape[0], self.H)))
        ).reshape(-1, self.V + self.H, 1)
        x_, u_masked_, target_mask = (
            torch.from_numpy(x).to(self.device).type(self.dtype),
            torch.from_numpy(u_masked).to(self.device).type(self.dtype),
            torch.from_numpy(1 - evidence_mask).to(self.device).type(self.dtype),
        )

        messages = self.infer(u_masked_, n_bp_iter, damping, beta, mode)
        beliefsV = (
            messages[0].sum(1, keepdim=True).transpose(1, 2)
            + self.b[: self.V]
            + u_masked_[:, : self.V]
        )[:, :, 0]
        beliefsV = beliefsV * strength
        loss = x_ * softplus(-beliefsV) + (1 - x_) * softplus(beliefsV)
        return (loss * target_mask).sum()

    def test(self, strength, n_bp_iter, p_target=None, x=None, masks=None):
        if x is None:
            x = self.X
        n_images = x.shape[0]
        mb_size = 125
        num, den = 0.0, 0.0
        for start in tqdm(range(0, n_images, mb_size)):
            stop = min(start + mb_size, n_images)
            x1 = x[start:stop]

            u = invsigmoid(x1).clip(-1000, 1000)
            if masks is not None:
                evidence_mask = 1 - masks[start:stop]
            else:
                evidence_mask = np.random.binomial(
                    1, 1.0 - p_target, u.shape
                )  # 0: target node, 1: evidence node,

            u_masked = np.hstack(
                (u * evidence_mask, np.zeros((u.shape[0], self.H)))
            ).reshape(-1, self.V + self.H, 1)
            u_ = torch.from_numpy(u_masked).to(self.device).type(self.dtype)

            N = u.shape[0]
            messages = torch.zeros(
                (2, N, self.H, self.V), device=self.device, dtype=self.dtype
            )
            for i in range(n_bp_iter):
                messages = layer(
                    messages,
                    self.Whv.detach(),
                    self.b.detach(),
                    u_,
                    mode='sum',
                    beta=1.0,
                )
            beliefsV = (
                (
                    (
                        messages[0].sum(1, keepdim=True).transpose(1, 2)
                        + self.b[: self.V]
                        + u_[:, : self.V]
                    )[:, :, 0]
                )
                .detach()
                .cpu()
                .numpy()
            )
            beliefsV *= strength
            logp = x1 * logsigmoid(beliefsV) + (1 - x1) * logsigmoid(-beliefsV)
            target_mask = 1 - evidence_mask
            logp *= target_mask
            num += -logp.sum() / np.log(2)
            den += target_mask.sum()
        return num / den


def layer(messages, Whv, b, u, mode, beta):
    """messages.shape == NxDxD, W.shape == DxD, b.shape == Dx1, u.shape == NxDx1
    W must have zero diagonal and be symmetric. Messages should have zero diagonal, but are not symmetric."""
    H, V = Whv.shape
    beliefsV = messages[0].sum(1, keepdim=True) + b[:V].t() + u[:, :V].transpose(1, 2)
    beliefsH = messages[1].sum(2, keepdim=True) + b[V:] + u[:, V:]

    incomingV = beliefsH - messages[1]
    incomingH = beliefsV - messages[0]

    new_messages = torch.empty_like(messages)
    new_messages[0] = propagate(incomingV, Whv, mode, beta)
    new_messages[1] = propagate(incomingH, Whv, mode, beta)
    return new_messages


def propagate(mess_in, w, mode="sum", beta=1, threshold=20):
    """Binary factor is (in logspace) -w if there's disagreement and 0 if there is agreement
    beta is a temperature parameter. 1 is sumprop, as it tends to inf, message prop tends to maxprop"""
    mess_out = w.sign() * torch.max(
        -w.abs(), torch.min(mess_in, w.abs())
    )  # maxprop equation
    if mode == "sum":
        # Refinement needed for sumprop. Both quantities in the subtraction are bounded by log(2), so this is very robust
        mess_out = (
            mess_out
            + (-(mess_in + w).abs()).exp().log1p()
            - (-(mess_in - w).abs()).exp().log1p()
        )  # alternative 1 (ignores beta)
        # mess_out = mess_out + softplus(-abs(mess_in + w), beta, threshold) - softplus(-abs(mess_in - w), beta, threshold)  # alternative 2 (uses beta)
        # mess_out = w + softplus(-(mess_in + w), beta, threshold) - softplus(w - mess_in, beta, threshold)  # alternative 3 (simpler, so maybe faster but less robust)
    elif mode != "max":
        assert False, "Unknown mode"
    return mess_out


def propagate_np(mess_in, W, mode="sum"):
    # Binary factor is (in logspace) -W if there's disagreement and 0 if there is agreement
    mess_out = np.sign(W) * mess_in.clip(-np.abs(W), np.abs(W))  # maxprop equation
    if mode == "sum":
        # Refinement needed for sumprop. Both quantities in the subtraction are bounded by log(2), so this is very robust
        mess_out += np.log1p(np.exp(-np.abs(mess_in + W))) - np.log1p(
            np.exp(-np.abs(mess_in - W))
        )
    elif mode != "max":
        assert False, "Unknown mode"
    return mess_out


def logsigmoid(x):
    return np.minimum(0, x) - np.log1p(np.exp(-np.abs(x)))
