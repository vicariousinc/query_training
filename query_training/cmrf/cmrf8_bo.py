""" Cloned MRF - linear messages"""

from __future__ import print_function

from builtins import range

import numba as nb
import numpy as np

import torch

torch.autograd.set_detect_anomaly(True)


class CMRF(object):
    def __init__(
        self,
        input_train,
        output_train,
        p_contour,
        n_clones,
        min_value_bu=0.0,
        dtype=torch.float32,
        device=None,
        requires_grad=True,
    ):
        assert (
            (input_train[:, 0, :] == 1).all()
            and (input_train[:, -1, :] == 1).all()
            and (input_train[:, :, 0] == 1).all()
            and (input_train[:, :, -1] == 1).all()
        ), "There's a contour touching the border"
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.n_clones = n_clones
        self.n_states = n_clones.sum()
        state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
        self.state_loc = torch.from_numpy(state_loc).to(self.device)
        self.logWtd = torch.randn(
            self.n_states,
            self.n_states,
            device=self.device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        self.logWlr = torch.randn(
            self.n_states,
            self.n_states,
            device=self.device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        self.logWfd = torch.randn(
            self.n_states,
            self.n_states,
            device=self.device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        self.logWsd = torch.randn(
            self.n_states,
            self.n_states,
            device=self.device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        self.images_bu = contours_to_mess_bu(input_train, p_contour, n_clones)
        self.images_target = img_to_mess_bu(output_train, n_clones)
        self.min_value_bu = min_value_bu
        self.neighbors = torch.zeros(input_train.shape[1:], dtype=self.dtype).to(
            self.device
        )
        self.neighbors[1:] += 1
        self.neighbors[:-1] += 1
        self.neighbors[:, 1:] += 1
        self.neighbors[:, :-1] += 1

    def train(
        self,
        learning_rate=1e-1,
        n_epochs=1,
        n_bp_iter=10,
        damping=0.5,
        mb_size=20,
        savetofile=None,
        save_every=100,
    ):
        n_images = self.images_bu.shape[0]
        optimizer = torch.optim.Adam(
            (self.logWtd, self.logWlr, self.logWfd, self.logWsd), lr=learning_rate
        )
        it = 0
        convergence = []
        for epoch in range(n_epochs):
            order = np.random.permutation(n_images)
            for start in range(0, n_images, mb_size):
                optimizer.zero_grad()
                stop = min(start + mb_size, n_images)
                images_bu = self.images_bu[order[start:stop]]
                images_target = self.images_target[order[start:stop]]
                loss_mb = 0.0
                num_bps_mb, den_bps_mb = 0.0, 0.0
                size_gpu_can_take = 25
                assert images_bu.shape[0] % size_gpu_can_take == 0
                for sample in range(images_bu.shape[0] // size_gpu_can_take):
                    images_bu_mb = images_bu[
                        sample * size_gpu_can_take : (sample + 1) * size_gpu_can_take
                    ]
                    images_target_mb = images_target[
                        sample * size_gpu_can_take : (sample + 1) * size_gpu_can_take
                    ]
                    logp, num_bps, den_bps = self.query_logp(
                        images_bu_mb, images_target_mb, n_bp_iter, damping
                    )
                    loss = -logp
                    loss_mb += loss.detach().cpu().numpy().item()
                    num_bps_mb += num_bps.detach().cpu().numpy().item()
                    den_bps_mb += den_bps
                    loss.backward()
                convergence.append(loss_mb / images_bu.shape[0])
                print(
                    "It:",
                    it,
                    ",",
                    "Ep:",
                    epoch,
                    ",",
                    start,
                    "/",
                    n_images,
                    convergence[-1],
                    num_bps_mb / den_bps_mb,
                )
                optimizer.step()
                if savetofile is not None and it % save_every == 0:
                    np.savez_compressed(
                        savetofile,
                        logWtd=self.logWtd.detach().cpu().numpy(),
                        logWlr=self.logWlr.detach().cpu().numpy(),
                        logWfd=self.logWfd.detach().cpu().numpy(),
                        logWsd=self.logWsd.detach().cpu().numpy(),
                        n_clones=self.n_clones,
                        convergence=np.array(convergence),
                        learning_rate=learning_rate,
                        damping=damping,
                        n_bp_iter=n_bp_iter,
                    )
                it += 1

    def load(self, filename):
        b = np.load(filename)
        assert (
            self.n_clones.shape[0] == b['n_clones'].shape[0]
            and (self.n_clones == b['n_clones']).all()
        ), "Number of clones don't match"
        self.logWtd.data[:] = torch.from_numpy(b['logWtd'])
        self.logWlr.data[:] = torch.from_numpy(b['logWlr'])
        self.logWfd.data[:] = torch.from_numpy(b['logWfd'])
        self.logWsd.data[:] = torch.from_numpy(b['logWsd'])

    def query_logp(self, images_bu, images_target, n_bp_iter, damping):
        targets = torch.from_numpy(images_target).to(self.device).type(self.dtype)
        evidences = torch.from_numpy(images_bu).to(self.device).type(self.dtype)
        n_targets = np.prod(images_bu.shape[:3])
        messages = self.infer(evidences, n_bp_iter, damping, "sum")
        logp = self.logprobsum(messages, targets)
        num_bps = -logp / np.log(2)
        den_bps = n_targets
        return logp, num_bps, den_bps

    def logprobsum(self, messages, targets):
        beliefs = messages.prod(4) + 1e-20
        beliefs /= beliefs.sum(3, keepdim=True)
        logp = (targets * beliefs).sum(3, keepdim=True).log().sum()
        return logp

    def test(self, n_bp_iter, damping, mode, images=None, mask=None, p_target=None):
        if images is None:
            images_bu = self.images_bu
        else:
            assert False
        if mask is None:
            targets, evidences, n_targets = self.split_images_bu(
                images_bu, p_target=p_target
            )
        else:
            targets, evidences, n_targets = self.split_images_bu(images_bu, mask=mask)
        beliefs = self.infer(
            evidences, n_bp_iter, damping, mode, return_belief_progress=True
        )
        return beliefs, targets

    def propagate_message(self, message, W, mode):
        n, h, w = message.size()[:3]
        if mode == "sum":
            new_message = (message.view(n, h, w, self.n_states, 1) * W).sum(3) + 1e-20
        elif mode == "max":
            new_message = (message.view(n, h, w, self.n_states, 1) * W).max(3)[
                0
            ] + 1e-20
        else:
            assert False, "Unknown mode"
        return new_message

    def incoming_messages(self, messages, evidences):
        # 405    0 up, 1 down, 2 left, 3 right
        # 2 3
        # 617
        diagonals = messages[:, :, :, :, 4:8].prod(4)
        # UP
        belief_except_down = (
            messages[:, :, :, :, 0]
            * messages[:, :, :, :, 2]
            * messages[:, :, :, :, 3]
            * diagonals
            * evidences
        )
        # DOWN
        belief_except_up = (
            messages[:, :, :, :, 1]
            * messages[:, :, :, :, 2]
            * messages[:, :, :, :, 3]
            * diagonals
            * evidences
        )
        # LEFT
        belief_except_right = (
            messages[:, :, :, :, 0]
            * messages[:, :, :, :, 1]
            * messages[:, :, :, :, 2]
            * diagonals
            * evidences
        )
        # RIGHT
        belief_except_left = (
            messages[:, :, :, :, 0]
            * messages[:, :, :, :, 1]
            * messages[:, :, :, :, 3]
            * diagonals
            * evidences
        )

        straight = messages[:, :, :, :, :4].prod(4)
        # 4
        belief_except_4 = (
            messages[:, :, :, :, 5]
            * messages[:, :, :, :, 6]
            * messages[:, :, :, :, 7]
            * straight
            * evidences
        )
        # 5
        belief_except_5 = (
            messages[:, :, :, :, 4]
            * messages[:, :, :, :, 6]
            * messages[:, :, :, :, 7]
            * straight
            * evidences
        )
        # 6
        belief_except_6 = (
            messages[:, :, :, :, 4]
            * messages[:, :, :, :, 5]
            * messages[:, :, :, :, 7]
            * straight
            * evidences
        )
        # 7
        belief_except_7 = (
            messages[:, :, :, :, 4]
            * messages[:, :, :, :, 5]
            * messages[:, :, :, :, 6]
            * straight
            * evidences
        )

        return (
            belief_except_down,
            belief_except_up,
            belief_except_right,
            belief_except_left,
            belief_except_4,
            belief_except_5,
            belief_except_6,
            belief_except_7,
        )

    def update_messages(self, messages, evidences, damping, mode):
        # Incoming messages
        (
            belief_except_down,
            belief_except_up,
            belief_except_right,
            belief_except_left,
            belief_except_4,
            belief_except_5,
            belief_except_6,
            belief_except_7,
        ) = self.incoming_messages(messages, evidences)

        # Propagate
        # 405    0 up, 1 down, 2 left, 3 right
        # 2 3
        # 617

        # UP
        message_up = self.propagate_message(belief_except_down, self.Wtd.t(), mode)
        # DOWN
        message_down = self.propagate_message(belief_except_up, self.Wtd, mode)
        # LEFT
        message_left = self.propagate_message(belief_except_right, self.Wlr.t(), mode)
        # RIGHT
        message_right = self.propagate_message(belief_except_left, self.Wlr, mode)

        # 4
        message_4 = self.propagate_message(belief_except_7, self.Wfd.t(), mode)
        # 5
        message_5 = self.propagate_message(belief_except_6, self.Wsd.t(), mode)
        # 6
        message_6 = self.propagate_message(belief_except_5, self.Wsd, mode)
        # 7
        message_7 = self.propagate_message(belief_except_4, self.Wfd, mode)

        # Assign
        new_messages = torch.ones(
            evidences.shape + (8,), device=self.device, dtype=self.dtype
        )  # 0 up, 1 down, 2 left, 3 right
        new_messages[:, :-1, :, :, 0] = (
            messages[:, :-1, :, :, 0] ** (1 - damping) * message_up[:, 1:, :] ** damping
        )
        new_messages[:, 1:, :, :, 1] = (
            messages[:, 1:, :, :, 1] ** (1 - damping)
            * message_down[:, :-1, :] ** damping
        )
        new_messages[:, :, :-1, :, 2] = (
            messages[:, :, :-1, :, 2] ** (1 - damping)
            * message_left[:, :, 1:] ** damping
        )
        new_messages[:, :, 1:, :, 3] = (
            messages[:, :, 1:, :, 3] ** (1 - damping)
            * message_right[:, :, :-1] ** damping
        )

        new_messages[:, :-1, :-1, :, 4] = (
            messages[:, :-1, :-1, :, 4] ** (1 - damping)
            * message_4[:, 1:, 1:] ** damping
        )
        new_messages[:, :-1, 1:, :, 5] = (
            messages[:, :-1, 1:, :, 5] ** (1 - damping)
            * message_5[:, 1:, :-1] ** damping
        )
        new_messages[:, 1:, :-1, :, 6] = (
            messages[:, 1:, :-1, :, 6] ** (1 - damping)
            * message_6[:, :-1, 1:] ** damping
        )
        new_messages[:, 1:, 1:, :, 7] = (
            messages[:, 1:, 1:, :, 7] ** (1 - damping)
            * message_7[:, :-1, :-1] ** damping
        )

        # Normalize
        if mode == "sum":
            new_messages = new_messages / new_messages.sum(3, keepdim=True)
        elif mode == "max":
            new_messages = new_messages / new_messages.max(3, keepdim=True)[0]
        else:
            raise AssertionError("Unknown mode")

        return new_messages

    def infer(
        self,
        evidences,
        n_bp_iter,
        damping,
        mode,
        return_belief_progress=False,
        detach=False,
    ):
        sum_from, sum_to = self.n_clones[0] + 1, self.n_clones[0] + 2
        if detach:  # faster when we don't need gradients
            self.Wtd = self.logWtd.detach().exp()
            self.Wlr = self.logWlr.detach().exp()
            self.Wfd = self.logWfd.detach().exp()
            self.Wsd = self.logWsd.detach().exp()
        else:
            self.Wtd = self.logWtd.exp()
            self.Wlr = self.logWlr.exp()
            self.Wfd = self.logWfd.exp()
            self.Wsd = self.logWsd.exp()
        messages = torch.ones(
            evidences.shape + (8,), device=self.device, dtype=self.dtype
        )  # 405    0 up, 1 down, 2 left, 3 right
        if return_belief_progress:  # 2 3
            beliefs = np.zeros((n_bp_iter,) + evidences.shape[:-1])  # 617
        for i in range(n_bp_iter):
            if return_belief_progress:
                b = (evidences * messages.prod(4)).detach()
                b /= b.sum(3, keepdim=True)
                # sl = self.state_loc.numpy()
                beliefs[i] = b[:, :, :, sum_from:sum_to].sum(3)
            messages = self.update_messages(messages, evidences, damping, mode)
        if return_belief_progress:
            return beliefs
        return messages


@nb.njit()
def contours_to_mess_bu(input_train, p_contour, n_clones):
    assert (input_train == 0).sum() + (
        input_train == 1
    ).sum() == input_train.size  # (binary inputs, where 0 is contour)
    assert len(p_contour) == 3
    n_states = n_clones.sum()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    messages_bu = np.zeros(input_train.shape + (n_states,))
    for i in range(input_train.shape[0]):
        for r in range(input_train.shape[1]):
            for c in range(input_train.shape[2]):
                v = input_train[i, r, c]
                for d in range(3):
                    start, stop = state_loc[d : d + 2]
                    messages_bu[i, r, c, start:stop] = (
                        p_contour[d] if v == 0 else 1 - p_contour[d]
                    )
    return messages_bu


@nb.njit()
def img_to_mess_bu(images, n_clones):
    n_states = n_clones.sum()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    messages_bu = np.zeros(images.shape + (n_states,))
    for i in range(images.shape[0]):
        for r in range(images.shape[1]):
            for c in range(images.shape[2]):
                v = images[i, r, c]
                start, stop = state_loc[v : v + 2]
                messages_bu[i, r, c, start:stop] = 1
    return messages_bu
