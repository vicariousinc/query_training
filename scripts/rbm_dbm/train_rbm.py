from __future__ import print_function

from builtins import range
from os import listdir, makedirs
from os.path import exists, join
from sys import argv

import numpy as np

import torch
from query_training.rbm_dbm.dataset import get_dataset
from query_training.rbm_dbm.rbm import RBM


def test_bits(dataset, basedirname='trained_models', result_dir='results', model='rbm'):
    basedirname += '_{}'.format(model)
    result_dir += '_{}'.format(model)
    # Get the RBM parameters from training
    assert exists(result_dir)
    dirname = join(basedirname, dataset)
    print(dirname)
    bp_bits, filenames, c = [], [], []
    _, _, x_test, n_hidden = get_dataset(dataset)
    for filename in listdir(dirname):
        if filename.split('.')[-1] == 'npz' and filename.split('.')[-1] == 'npz':
            print(filename)
            npz = np.load(join(dirname, filename))
            convergence, Whv, b, strength = (
                npz['convergence'],
                npz['Whv'],
                npz['b'],
                npz['strength'],
            )
            c.append(len(convergence))
            rbm = RBM(x_test, H=n_hidden)
            rbm.Whv[:] = torch.from_numpy(Whv)
            rbm.b[:] = torch.from_numpy(b)
            bp_bits.append(rbm.test(strength=strength, n_bp_iter=10, p_target=0.5))
            filenames.append(filename)
            np.savez_compressed(
                join(result_dir, dataset + '_qtnn_bp_test.npz'),
                convergence=np.array(c),
                bp_bits=np.array(bp_bits),
                filenames=np.array(filenames),
            )


def train_rbm(x_train, x_val, n_hidden, lr, n_epochs, savedir, n_iter=1000):
    assert n_iter > 100  # we checkpoint every 100 iterations
    rbm = RBM(x_train, H=n_hidden)
    return rbm.train(
        x_val,
        learning_rate=lr,
        n_epochs=n_epochs,
        n_bp_iter=10,
        damping=1.0,
        beta=1.0,
        mb_size=500,
        p_target=0.5,
        save_every=1,
        savetofile=join(savedir, 'checkpoint'),
    )


def train_all_lr(dataset, n_epochs, lr, basedirname='trained_models', model='rbm'):
    basedirname += '_{}'.format(model)
    savedir = join(basedirname, dataset)
    if not exists(savedir):
        makedirs(savedir)

    x_train, x_val, _, n_hidden = get_dataset(dataset)
    if model == 'dbm':
        x_hidden_train = 0.5 * np.ones((x_train.shape[0], n_hidden))
        x_hidden_val = 0.5 * np.ones((x_val.shape[0], n_hidden))
        x_train = np.concatenate((x_train, x_hidden_train), axis=1)
        x_val = np.concatenate((x_val, x_hidden_val), axis=1)
    else:
        assert model == 'rbm', 'Unsupported model {}'.format(model)

    print(dataset, n_hidden, x_train.shape[1])
    for realization in range(2, 5):
        print("Realization {}, LR {}".format(realization, lr))
        convergence, convergence_val, best_rbm, bits = train_rbm(
            x_train, x_val, n_hidden, lr, n_epochs, savedir=savedir, n_iter=1000
        )
        np.savez_compressed(
            join(savedir, 'rbm_{}_lr{}_r{}'.format(n_epochs, lr, realization)),
            convergence=convergence,
            convergence_val=convergence_val,
            Whv=best_rbm[0],
            b=best_rbm[1],
            strength=best_rbm[2],
            bits=bits,
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
    learning_rates = {
        'adult': 0.01,
        'connect4': 0.003,
        'digits': 0.003,
        'dna': 0.01,
        'mushrooms': 0.003,
        'nips': 0.003,
        'ocr_letters': 0.003,
        'rcv1': 0.003,
        'web': 0.003,
    }
    n_epochs_all = {
        'adult': 1000,
        'connect4': 1000,
        'digits': 1000,
        'dna': 1000,
        'mushrooms': 1000,
        'nips': 200,
        'ocr_letters': 1000,
        'rcv1': 1000,
        'web': 1000,
    }
    if argv[1] == 'train':
        dataset = datasets[int(argv[3])]
        train_all_lr(
            dataset, n_epochs_all[dataset], learning_rates[dataset], model=argv[2]
        )
    elif argv[1] == 'test':
        for dataset in datasets:
            test_bits(dataset, model=argv[2])
    else:
        print("choose train, test, then the dataset number")
