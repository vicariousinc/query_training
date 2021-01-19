from __future__ import print_function

from builtins import range
from os import makedirs
from os.path import exists, join
from sys import argv

import numpy as np
from sklearn.neural_network import BernoulliRBM

import torch
from query_training.rbm_dbm.dataset import get_dataset
from query_training.rbm_dbm.rbm import RBM


def test_rbm_pcd_gibbs(
    x_test, Whv, bh, bv, p_target=0.5, n_gibbs_steps=5000, thinning=10, burnin=20
):
    rbm = BernoulliRBM(n_components=Whv.shape[0], learning_rate=0.0)
    rbm.components_, rbm.intercept_hidden_, rbm.intercept_visible_ = Whv, bh, bv
    evidence_mask = np.random.binomial(
        1, p_target, x_test.shape
    )  # 0: target node, 1: evidence node,

    V = np.random.binomial(1, p_target, x_test.shape)
    V = x_test * evidence_mask + V * (1 - evidence_mask)
    prob1 = np.zeros_like(V)
    count = 0
    for it in range(n_gibbs_steps):
        V = rbm.gibbs(V)
        V = x_test * evidence_mask + V * (1 - evidence_mask)
        if (it + 1) % thinning == 0 and it > burnin:
            prob1 += V
            count += 1
    prob1 /= count
    prob1_clipped = prob1.clip(1e-15, 1 - 1e-15)
    target_mask = 1 - evidence_mask
    logp = x_test * np.log(prob1_clipped) + (1 - x_test) * np.log(1 - prob1_clipped)
    logp *= target_mask
    return -logp.sum() / target_mask.sum() / np.log(2)


def test_rbm_pcd(X, Whv, bh, bv, inftype):
    if inftype == 'gibbs':
        # gibbs scoring
        return test_rbm_pcd_gibbs(
            X, Whv, bh, bv, p_target=0.5, n_gibbs_steps=5000, thinning=10, burnin=20
        )
    elif inftype == 'bp':
        # bp scoring
        Whv = Whv / 2
        bv = Whv.sum(0) + bv
        bh = Whv.sum(1) + bh
        b = np.hstack((bv, bh)).reshape(-1, 1)

        rbm = RBM(X, H=Whv.shape[0])
        rbm.Whv[:] = torch.from_numpy(Whv)
        rbm.b[:] = torch.from_numpy(b)

        return rbm.test(strength=1.0, n_bp_iter=10, p_target=0.5)
    else:
        assert False, "use inftype gibbs or bp"


def train_rbm_pcd(x_train, x_val, n_hidden, lr, inftype, n_iter=1000):
    assert n_iter > 100  # we checkpoint every 100 iterations
    rbm = BernoulliRBM(
        n_components=n_hidden,
        learning_rate=lr,
        batch_size=x_train.shape[0],
        n_iter=n_iter,
        verbose=0,
    )
    best_score, best_rbm = np.inf, None
    for it in range(n_iter):
        rbm.partial_fit(x_train)
        if (it + 1) % 20 == 0:  # checkpoint every 20
            score = test_rbm_pcd(
                x_val,
                rbm.components_,
                rbm.intercept_hidden_,
                rbm.intercept_visible_,
                inftype,
            )
            if score < best_score:
                best_score = score
                best_rbm = (
                    rbm.components_.copy(),
                    rbm.intercept_hidden_.copy(),
                    rbm.intercept_visible_.copy(),
                )
    return best_rbm, best_score


def train_all_lr_pcd(dataset, inftype, basedirname='pcd_trained_rbms'):
    savedir = join(basedirname, dataset)
    if not exists(savedir):
        makedirs(savedir)
    x_train, x_val, _, n_hidden = get_dataset(dataset)
    print(dataset, n_hidden, x_train.shape[1])
    for realization in range(5):
        for lr in [0.03, 0.1, 0.3, 1, 3]:
            print("Realization {}, LR {}".format(realization, lr))
            (Whv, bh, bv), bits = train_rbm_pcd(
                x_train, x_val, n_hidden, lr, inftype, n_iter=1000
            )
            np.savez_compressed(
                join(savedir, 'rbm_{}_lr{}_r{}'.format(inftype, lr, realization)),
                Whv=Whv,
                bh=bh,
                bv=bv,
                bits=bits,
            )


def choose_lr(dataset, inftype, basedirname):
    dirname = join(basedirname, dataset)
    lrs = [0.03, 0.1, 0.3, 1, 3]
    bits = []
    for lr in lrs:
        bits_ = []
        for realization in range(5):
            filename = 'rbm_{}_lr{}_r{}.npz'.format(inftype, lr, realization)
            bits_.append(np.load(join(dirname, filename))['bits'])
        bits.append(np.array(bits_).mean())
    lr = lrs[np.array(bits).argmin()]
    print(bits, lr)
    return lr


def test_all_lr_pcd(
    dataset, inftype, basedirname='pcd_trained_rbms', result_dir='results'
):
    # Get the RBM parameters from training
    assert exists(result_dir)
    dirname = join(basedirname, dataset)
    print(dirname)
    bits, filenames = [], []
    _, _, x_test, n_hidden = get_dataset(dataset)
    lr = choose_lr(
        dataset, 'bp', basedirname
    )  # force this for the lr selection because gibbs takes forever to run
    for realization in range(5):
        filename = 'rbm_{}_lr{}_r{}.npz'.format(
            'bp', lr, realization
        )  # force this for the lr selection because gibbs takes forever to run
        print(filename)
        npz = np.load(join(dirname, filename))
        Whv, bh, bv = npz['Whv'], npz['bh'], npz['bv']
        bits.append(test_rbm_pcd(x_test, Whv, bh, bv, inftype))
        filenames.append(filename)
        np.savez_compressed(
            join(result_dir, dataset + '_pcd_{}_test.npz'.format(inftype)),
            bits=np.array(bits),
            filenames=np.array(filenames),
        )


if __name__ == '__main__':
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
    dataset = datasets[int(argv[1])]
    train_all_lr_pcd(dataset, 'bp')
    # train_all_lr_pcd(dataset, 'gibbs')
    test_all_lr_pcd(dataset, 'bp')
    test_all_lr_pcd(dataset, 'gibbs')
