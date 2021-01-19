import os

import joblib

from query_training import BASE
from query_training.grbm.grbm_qt import GRBM_QT

LOAD_SAVED_MODEL = True
BOX_EXPERIMENT = False
BASE = os.path.join(BASE, 'grbm')

dataset = joblib.load(os.path.join(BASE, 'frey_faces.joblib'))
X_train = dataset['train'][0]
X_test = dataset['test'][0]

grbm = GRBM_QT(X_train, 200)

if LOAD_SAVED_MODEL:
    # Load pre-trained model from file.
    grbm.load_from_file(os.path.join(BASE, 'qt', 'qtnn_params.npz'))
else:
    # Train model.
    grbm.train()


if BOX_EXPERIMENT:
    print("Running box experiment...")
    masks = joblib.load(os.path.join(BASE, 'frey_faces_box_test_masks.joblib'))
else:
    print("Running random mask experiment...")
    masks = joblib.load(os.path.join(BASE, 'frey_faces_test_masks.joblib'))
    masks = 1.0 - masks

loss = grbm.evaluate(test_X=X_test, evidence_masks=masks, n_bp_iter=50, batch_size=5)
print("NCE: {}".format(loss))
