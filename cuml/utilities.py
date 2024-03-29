import numpy as np
import os
import pickle
import sklearn

'''
Contains any utility functions needed for the rf_classifier.py script
'''


# Save objects to pickle file given list of objects and file name WITHOUT file extension
def save_objects(object_list, filename):
    with open(filename+".pkl", 'wb') as f:
        pickle.dump(object_list, f)


# Load objects from .pkl file, provided name of pickle file WITHOUT file extension
def load_objects(filename):
    with open(filename+".pkl", 'rb') as f:
        return pickle.load(f)

### EXAMPLE of save and load
# a, b = 10, 11
# save_objects([a,b], "test_save")

# c,d = load_objects("test_save")

# print(c)
# print(d)


# Load csr lists from text file (PROVIDE EXTENSION)
def loadcsr_from_txt(filename):
    ret_map = {}
    file = open(filename, "r")
    i = 0
    lines = file.readlines()
    for i in range(0,len(lines),3):
        values = lines[i+2].split(", ")
        values = values[0:len(values)-1]

        ret_map[lines[i][0:len(lines[i])-1]] = [float(j) for j in values]
    return ret_map

# print(loadcsr_from_txt("td7_csr.txt"))