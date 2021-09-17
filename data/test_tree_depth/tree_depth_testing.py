import numpy as np
import os
import time
import pandas as pd

# test classification dataset
from sklearn.datasets import make_classification
# make predictions using random forest for classification
from sklearn.ensemble import RandomForestClassifier
# split data into train and test
from sklearn.model_selection import train_test_split

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt

import scipy
# from libsvm.svmutil import *

# define dataset
# X_all, y_all = make_classification(n_samples=_n_samples, n_features=_n_features, n_informative=_n_informative, n_redundant=_n_redundant, random_state=_random_state)
DATASET_PATH = "all_train"
DATASET_NAME = "HIGGS"

_n_samples= 1100000
_n_features= 28
_n_redundant= 3
_n_informative = _n_features - _n_redundant 
_random_state= 4

# define dataset
x_all, y_all = make_classification(n_samples=_n_samples, n_features=_n_features, n_informative=_n_informative, n_redundant=_n_redundant, random_state=_random_state)

# print("Reading")
# y_all, x_all = svm_read_problem(DATASET_PATH, return_scipy = True)
# print("Read")
# x_all = np.asarray(x_all)
# df = pd.read_csv(DATASET_PATH+".csv")
# y_ids = df.iloc[:, :1]
# x_pd = df.iloc[:, 1:-1]
# print(y_ids)

# # meta_info = pd.read_csv("./gas/HT_sensor_metadata_info.csv")
# x_all = x_pd.to_numpy()

# y_all = y_ids.to_numpy().ravel()


# for i in range(len(y_all)):
#     if int(y_all[i]) != 0:
#         y_all[i] = 1
#     else:
#         y_all[i] = 0
    # print(int(y_ids.iloc[i]))
    # ind = int(y_ids.iloc[i])
    # y_all[i] = meta_info.iloc[ind, 1]
    # print(y_all[i])

# y_all = y_all.to_numpy().ravel()

# for i in range(x_all.shape[0]):
#     for j in range(x_all.shape[1]):
#         if np.isnan(x_all[i,j]):
#             x_all[i,j] = 0







# X_use, X_discard, y_use, y_discard = train_test_split(x_all, y_all, test_size=0.85)
X, X_test, y, y_test = train_test_split(x_all, y_all, test_size=0.2)
print("training size= y: "+str(y.shape) +"  x: "+str(X.shape))
def write_array(arr_name,str_name,f):
    f.write("{}\n".format(str_name))
    f.write("{0},\n".format(len(arr_name)))
    for val in arr_name:
        f.write("{0}, ".format(val))
    f.write("\n")



save_objects([X_test, y_test], "TESTSET"+DATASET_NAME)

#configure the forest
_n_estimators = 50 
MAX_DEPTH = 60 
MAX_ESTIMATORS = 100
FIXED_DEPTH = 45
#_subtree_depth =  TO BE CONFIGURED IN BELOW 

depth_scores = []
for _max_depth in range(10, MAX_DEPTH+1, 5):    
    
    # define the model
    model = RandomForestClassifier(n_estimators= _n_estimators, max_depth = _max_depth)
    
    # fit the model on the whole dataset
    print("Training the model with "+str(_max_depth)+" max depth")
    start_time = time.time()
    model.fit(X, y)
    print(str(time.time() - start_time)+" seconds to train")

    average_score = model.score(X_test, y_test)
    print("Score: "+str(average_score))
    depth_scores.append(average_score)
    save_objects([model], "MODEL"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators))

estimators_scores=[]

for estimators in range(10, MAX_ESTIMATORS+1, 10):    
    
    # define the model
    model = RandomForestClassifier(n_estimators= estimators, max_depth = FIXED_DEPTH)
    
    # fit the model on the whole dataset
    print("Training the model with "+str(estimators)+" trees")
    start_time = time.time()
    model.fit(X, y)
    print(str(time.time() - start_time)+" seconds to train")

    average_score = model.score(X_test, y_test)
    print("Score: "+str(average_score))
    estimators_scores.append(average_score)
    save_objects([model], "MODEL"+DATASET_NAME+"_td"+str(FIXED_DEPTH)+"_ne"+str(estimators))

# with open("depthscore_"+DATASET_NAME+".txt", "w") as f:
#     write_array(depth_scores, "Average scores", f)

with open("estimatorscore_"+DATASET_NAME+".txt", "w") as f:
    write_array(estimators_scores, "Average scores", f)
