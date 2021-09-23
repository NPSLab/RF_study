from data.synthetic.rf_layout import MAX_ESTIMATORS
from pickle import load
import sklearn, sklearn.datasets, numpy as np
from numba import cuda
from cuml import ForestInference
import os

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt



DIRNAME = os.path.dirname(__file__)

MODEL_PATH = "data\test_tree_depth\Covtype_trained"
DATASET_NAME = "covtype"

TESTSET_PATH = r''
MAXDEPTH = 60
MAXESTIMATORS = 100
FIXED_DEPTH = 45
FIXED_ESTIMATORS = 50

X_test, y_test = load_objects(TESTSET_PATH)

for depth in range(10, MAXDEPTH + 1, 5):
    for estimators in range(10, MAX_ESTIMATORS+1, 10):
        model = load_objects(os.path.join(DIRNAME, MODEL_PATH, "MODEL"+DATASET_NAME+"_td"+str(depth)+"_ne"+str(estimators)))[0]

        X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
        fm = ForestInference.load_from_sklearn(model, output_class=True)
        fil_preds_gpu = fm.predict(X_gpu)
        accuracy_score = sklearn.metrics.accuracy_score(y_test,
                    np.asarray(fil_preds_gpu))

