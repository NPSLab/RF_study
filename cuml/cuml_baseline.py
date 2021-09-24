# from data.synthetic.rf_layout import MAX_ESTIMATORS
from pickle import load
import sklearn, sklearn.datasets, numpy as np
from numba import cuda
from cuml import ForestInference
import os
import time

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt



DIRNAME = os.path.dirname(__file__)

# MODEL_PATH = "data\test_tree_depth\Covtype_trained"
DATASET_NAME = ["HIGGS"]
DEPTHS = [25, 45, 45]
NUM_ESTIMATORS = [50, 50, 100]

for i in range(len(DATASET_NAME)):
    testset_name = "TESTSET"+DATASET_NAME[i]

    X_test, y_test = load_objects(testset_name)

    results_file = open(DATASET_NAME[i]+"_cuml_results.txt", 'w')

    for j in range(len(DEPTHS)):
        results_file.write("td: " +str(DEPTHS[i])+" ne: "+str(NUM_ESTIMATORS[i])+ "\n")
        model = load_objects("MODEL"+DATASET_NAME[i]+"_td"+str(DEPTHS[j])+"_ne"+str(NUM_ESTIMATORS[j]))[0]

        X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
        
        fm = ForestInference.load_from_sklearn(model, output_class=True)
        start_time = time.time()
        fil_preds_gpu = fm.predict(X_gpu)
        end_time = time.time()
        accuracy_score = sklearn.metrics.accuracy_score(y_test, np.asarray(fil_preds_gpu))
        results_file.write("Average Accuracy: "+str(accuracy_score)+"\n")
        results_file.write("Time to complete: "+str(end_time)+"\n")
        results_file.write("\n")
    results_file.close()
        

