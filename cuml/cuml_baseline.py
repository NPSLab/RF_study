# from data.synthetic.rf_layout import MAX_ESTIMATORS
# from data.test_tree_depth.tree_depth_testing import DATASET_PATH
from pickle import load
import sklearn, sklearn.datasets, numpy as np
from numba import cuda
from cuml import ForestInference
import os
import time

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt
from libsvm.svmutil import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



DIRNAME = os.path.dirname(__file__)

# MODEL_PATH = "data\test_tree_depth\Covtype_trained"
DATASET_PATH = "covtype.libsvm.binary"
DATASET_NAME = ["HIGGS", "SUSY"]
DEPTHS = [20, 40, 60, 45, 45, 45]
NUM_ESTIMATORS = [50, 50, 50, 10, 50, 100]

# y_all, x_all = svm_read_problem(DATASET_PATH, return_scipy = True)
# x_all = x_all.toarray()
# y_all = y_all.to_numpy().ravel()

# X, X_test, y, y_test = train_test_split(x_all, y_all, test_size=0.2)

for i in range(len(DATASET_NAME)):
    testset_name = "TESTSET"+DATASET_NAME[i]

    X_test, y_test = load_objects(testset_name)

    results_file = open(DATASET_NAME[i]+"_cuml_results.txt", 'w')

    for j in range(len(DEPTHS)):
        print("Running "+"td: " +str(DEPTHS[j])+" ne: "+str(NUM_ESTIMATORS[j]))
        results_file.write("td: " +str(DEPTHS[j])+" ne: "+str(NUM_ESTIMATORS[j])+ "\n")
        model = load_objects("MODEL"+DATASET_NAME[i]+"_td"+str(DEPTHS[j])+"_ne"+str(NUM_ESTIMATORS[j]))[0]
        # model = RandomForestClassifier(n_estimators= NUM_ESTIMATORS[j], max_depth = DEPTHS[j])
        # model.fit(X, y)
        # preds = model.predict(X_test)
        # score = sklearn.metrics.accuracy_score(y_test, preds)
        # results_file.write("Expected Score: "+str(score) +"\n")
        X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
        
        fm = ForestInference.load_from_sklearn(model, output_class=True)
        start_time = time.time()
        fil_preds_gpu = fm.predict(X_gpu)
        end_time = time.time() - start_time
        accuracy_score = sklearn.metrics.accuracy_score(y_test, np.asarray(fil_preds_gpu))
        results_file.write("Average Accuracy: "+str(accuracy_score)+"\n")
        results_file.write("Time to complete: "+str(end_time)+"\n")
        results_file.write("\n")
    results_file.close()
        

