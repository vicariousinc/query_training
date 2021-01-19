from __future__ import print_function

import numpy as np

import torch

torch.autograd.set_detect_anomaly(True)


class GRBM_QT(object):
    """
    Parameters
    ----------
    X : (num_images, num_visible_units) np.ndarray
        Data
    H : int
        Number of hidden units. Must be > 0.
    """

    def __init__(self, X, H, W=None):
        self.X = X
        self.H = H
        self.V = self.X.shape[1]

        self.dtype = torch.float32
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pairwise factor.
        self.W = torch.randn(
            self.H, self.V, device=self.dev, dtype=self.dtype, requires_grad=True
        )
        self.W.data *= 0.05

        # Unary factor on hidden.
        self.c = torch.zeros(
            self.H, 1, device=self.dev, dtype=self.dtype, requires_grad=True
        )

        # Gaussian parameters (natural) of unary factor on visible.
        self.mu = torch.zeros(
            self.V, 1, device=self.dev, dtype=self.dtype, requires_grad=True
        )
        self.log_var = torch.ones(
            self.V, 1, device=self.dev, dtype=self.dtype, requires_grad=True
        )
        self.log_var.data[:] = np.log(1.0)

    def generate_unary_visible(self, x, evidence_mask, observed_var=1e-3):
        x_ = torch.from_numpy(x).to(self.dev).type(self.dtype)

        target_mask_ = torch.from_numpy(1 - evidence_mask).to(self.dev).type(self.dtype)

        # Mean is the observed value for those that are observed.
        means = self.mu.t() * target_mask_ + x_ * (1.0 - target_mask_)
        variances = (torch.exp(self.log_var.t()) * target_mask_) + (
            observed_var * (1.0 - target_mask_)
        )

        unary_visible_ = get_natural_from_mean_var(torch.stack((means, variances), -1))

        return x_, target_mask_, unary_visible_

    def train(
        self,
        num_epochs=100,
        n_bp_iter=50,
        mb_size=200,
        sb_size=20,
        p_target=0.5,
        box_mask=False,
    ):
        n_images = self.X.shape[0]

        optimizer = torch.optim.Adam((self.W, self.c, self.mu), lr=5e-3)

        for epoch in range(num_epochs):
            print("Epoch: {}".format(epoch))
            random_order = np.random.permutation(n_images)

            for start in range(0, n_images, mb_size):

                optimizer.zero_grad()
                loss_mb = 0.0

                # Minibatch
                X_mb = self.X[random_order[start : min(start + mb_size, n_images)]]

                for sample in range(int(X_mb.shape[0] / sb_size)):

                    X_sb = X_mb[sample * sb_size : (sample + 1) * sb_size]

                    # Generate a random mask
                    # 0: target node  1: evidence node,
                    evidence_mask = np.random.binomial(1, 1.0 - p_target, X_sb.shape)

                    x, target_mask, unary_visible = self.generate_unary_visible(
                        x=X_sb, evidence_mask=evidence_mask
                    )

                    # Infer mean and variance of visible units.
                    belief = self.infer(target_mask, unary_visible, n_bp_iter=n_bp_iter)
                    belief_mean = belief[..., 0]
                    belief_var = belief[..., 1]

                    loss = -(
                        (x * belief_mean)
                        + (x ** 2 * belief_var)
                        + belief_mean ** 2 / (4.0 * belief_var)
                        + 0.5 * torch.log(-2.0 * belief_var)
                    )
                    loss = (loss * target_mask).sum()
                    loss.backward()
                    print(loss)

                    loss_mb += loss.detach().cpu().numpy().item()

                optimizer.step()

            self.save_to_file("{}_epoch_{}_bp.npz".format(epoch, n_bp_iter))

    def infer(self, target_mask, unary_visible, n_bp_iter=15):
        # Messages from visible to hidden. Logit.
        self.messages_VH = torch.zeros(
            (unary_visible.shape[0], self.H, self.V), device=self.dev, dtype=self.dtype
        )

        # Messages from hidden to visible. Gaussian natural parametrization.
        self.messages_HV = torch.zeros(
            (unary_visible.shape[0], self.H, self.V, 2),
            device=self.dev,
            dtype=self.dtype,
        )
        self.messages_HV.data[..., 1] = -0.5

        for i in range(n_bp_iter):
            belief_V = self.messages_HV.sum(1) + unary_visible
            cavity_V = belief_V.unsqueeze(1) - self.messages_HV

            self.messages_VH = -(2.0 * cavity_V[..., 0] * self.W + self.W ** 2) / (
                4 * cavity_V[..., 1]
            )

            belief_H = self.messages_VH.sum(2, keepdim=True) + self.c
            cavity_H = belief_H - self.messages_VH

            self.messages_HV = self.update_message_approximations(
                cavity_V, cavity_H
            ) * target_mask.unsqueeze(1).unsqueeze(-1)

            del belief_V
            del cavity_V
            del belief_H
            del cavity_H

        belief_V = self.messages_HV.sum(1) + unary_visible
        return belief_V

    def update_message_approximations(self, cavity_V, cavity_H):
        mean_var = get_mean_var_from_natural(cavity_V)
        mu = mean_var[..., 0]
        var = mean_var[..., 1]

        C = torch.exp(cavity_H + self.W * mu + 0.5 * var * self.W ** 2)
        approx_mean = C * (mu + var * self.W) / (C + 1.0) + 1.0 * mu / (C + 1.0)

        expectation_x2 = C * ((mu + var * self.W) ** 2 + var) / (C + 1.0) + 1.0 * (
            mu ** 2 + var
        ) / (C + 1.0)

        approx_var = expectation_x2 - approx_mean ** 2

        approx_natural = get_natural_from_mean_var(
            torch.stack((approx_mean, approx_var), -1)
        )
        return approx_natural - cavity_V

    def evaluate(self, test_X, evidence_masks, n_bp_iter, batch_size):
        total_loss = 0.0
        n_images = test_X.shape[0]
        for start in range(0, n_images, batch_size):
            X_mb = test_X[start : min(start + batch_size, n_images)]
            evidence_mask = evidence_masks[start : min(start + batch_size, n_images)]

            x, target_mask, unary_visible = self.generate_unary_visible(
                x=X_mb, evidence_mask=evidence_mask
            )

            # Infer mean and variance of visible units.
            belief = self.infer(target_mask, unary_visible, n_bp_iter=n_bp_iter)
            belief_mean = belief[..., 0]
            belief_var = belief[..., 1]

            loss = np.log(1.0 / np.sqrt(2.0 * np.pi)) + (
                (x * belief_mean)
                + (x ** 2 * belief_var)
                + (belief_mean ** 2) / (4.0 * belief_var)
                + 0.5 * torch.log(-2.0 * belief_var)
            )

            for i in range(loss.shape[0]):
                image_loss = -(
                    loss[i][target_mask[i] == 1.0].mean().detach().cpu().numpy().item()
                )
                total_loss += image_loss

        return total_loss / n_images

    def save_to_file(self, filename):
        np.savez_compressed(
            filename,
            W=self.W.detach().cpu().numpy(),
            c=self.c.detach().cpu().numpy(),
            mu=self.mu.detach().cpu().numpy(),
            log_var=self.log_var.detach().cpu().numpy(),
        )

    def load_from_file(self, filename):
        data = np.load(filename)
        self.W.data[:] = torch.from_numpy(data["W"])
        self.c.data[:] = torch.from_numpy(data["c"])
        self.mu.data[:] = torch.from_numpy(data["mu"])
        self.log_var.data[:] = torch.from_numpy(data["log_var"])


def get_mean_var_from_natural(X):
    out = torch.stack((-X[..., 0] / 2.0 / X[..., 1], -1.0 / 2.0 / X[..., 1]), -1)
    return out


def get_natural_from_mean_var(X):
    out = torch.stack((X[..., 0] / X[..., 1], -1.0 / 2.0 / X[..., 1]), -1)
    return out
