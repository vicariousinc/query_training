import subprocess

for ii in range(9):
    subprocess.call(['python', 'train_rbm.py', 'train', 'rbm', str(ii)])

for ii in range(9):
    subprocess.call(['python', 'train_rbm.py', 'train', 'dbm', str(ii)])
