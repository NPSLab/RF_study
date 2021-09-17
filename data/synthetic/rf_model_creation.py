import numpy as np
import os

# test classification dataset
from sklearn.datasets import make_classification
# make predictions using random forest for classification
from sklearn.ensemble import RandomForestClassifier
# split data into train and test
from sklearn.model_selection import train_test_split

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt

#configure the dataset
_n_samples=581012*2
_n_features=54
_n_redundant=5
_n_informative= _n_features - _n_redundant 
_random_state=3

# define dataset
X_all, y_all = make_classification(n_samples=_n_samples, n_features=_n_features, n_informative=_n_informative, n_redundant=_n_redundant, random_state=_random_state)

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.5)

def write_array(arr_name,str_name,f):
    f.write("{}\n".format(str_name))
    f.write("{0},\n".format(len(arr_name)))
    for val in arr_name:
        f.write("{0}, ".format(val))
    f.write("\n")

def write_2darray(arr,str_name,f):
    num_rows,num_cols = arr.shape
    f.write("{}\n".format(str_name))
    f.write("{0}, {1},\n".format(num_rows,num_cols))
    for row in arr:
        for col in row:
            f.write("{0}, ".format(col))
    f.write("\n")

print("save test inputs")
with open("test_input.txt",'w') as f:
    ##X_test, y_test 
    write_2darray(X_test,"X_test",f)
    write_array(y_test,"y_test",f)

#configure the forest
_n_estimators = 100 
_max_depth = 15 
#_subtree_depth =  TO BE CONFIGURED IN BELOW 

for _max_depth in range(7,16):    
    
    # define the model
    model = RandomForestClassifier(n_estimators= _n_estimators, max_depth = _max_depth)
    
    # fit the model on the whole dataset
    print("Training the model")
    model.fit(X, y)
    
    #generate strings for each feature name used in dot file, feature[i] is str(i)
    feature_list = [str(i) for i in range(0,_n_features)]
    
    csr_forest_trees = []
    
    print("Start transforming the tree")
    
    for idx in range(0,_n_estimators):
        print("Transforming tree:{0}".format(idx))
        curr_tree = model.estimators_[idx].tree_
        num_of_nodes = curr_tree.node_count 
    
        print("Start building connection matrix")
        
        print("Done building connection matrix")
        #CSR format for a tree
        print("Start building CSR format")
        #index of each node into edge list 
        node_list = np.zeros(num_of_nodes+1,dtype='i') 
        #reserve 2*N spaces, in fact, will be less than 2N, stores all edges
        edge_list = np.zeros(2*num_of_nodes,dtype='i') 
        #indicate of node is leaf, redundant, can be deducted from node_list, if node_list[i+1] = node_list[i], then node i is leaf node
        node_is_leaf = np.zeros(num_of_nodes,dtype='i') 
        curr_idx = 0
    
        for i in range(num_of_nodes):
            node_list[i] = curr_idx
            found_one = 0
    
            if curr_tree.children_left[i] != -1:
                edge_list[curr_idx] = curr_tree.children_left[i]
                curr_idx+=1
                found_one = 1
    
            if curr_tree.children_right[i] != -1:
                edge_list[curr_idx] = curr_tree.children_right[i]
                curr_idx+=1
                found_one = 1
    
            if found_one == 0:
                node_is_leaf[i]=1
        
        node_list[num_of_nodes] = curr_idx
        edge_list.resize(curr_idx)
        
        # read feature number and value used in each node
        node_values = np.zeros(num_of_nodes) 
        
        node_features = np.zeros(num_of_nodes, dtype='i') 
        
        for i in range(num_of_nodes):
            #mata_string = graph.get_node(str(i))[0].get_label()
            if node_is_leaf[i]:
                if model.estimators_[idx].tree_.value[i][0][0] >= model.estimators_[idx].tree_.value[i][0][1] : 
                    node_values[i]=0
                else:
                    node_values[i]=1
            else:
                node_features[i] = model.estimators_[idx].tree_.feature[i] 
                node_values[i] = model.estimators_[idx].tree_.threshold[i] 
        
        print("Done building CSR tree")
    
        #NOW These arrays are ready for the current tree/estimator X
        #num_of_nodes
        #node_list
        #edge_list
        #node_is_leaf
        #node_features
        #node_values
    
        csr_decision_tree = [num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values]
        csr_forest_trees.append(csr_decision_tree)
        print("Add current CSR tree into forest")
    
    print("\n\n Now we are trying to write CSR trees layouts")
    num_of_trees = len(csr_forest_trees) 
    
    if num_of_trees != _n_estimators:
        print("error")
    
    node_list_idx      = np.zeros(num_of_trees+1,dtype='i')
    edge_list_idx      = np.zeros(num_of_trees+1,dtype='i')
    node_is_leaf_idx   = np.zeros(num_of_trees+1,dtype='i')
    node_features_idx  = np.zeros(num_of_trees+1,dtype='i')
    node_values_idx    = np.zeros(num_of_trees+1,dtype='i')
    
    node_list_total = []
    edge_list_total = []
    node_is_leaf_total =[]
    node_features_total = []
    node_values_total = []
    #consolidate trees into a signle array
    def consolidate_csr(arr,arr_idx,csr_forest_trees,element):
        num_of_trees=len(csr_forest_trees)
        idx = 0
        for i in range(num_of_trees):
            arr_idx[i] = idx 
            arr.extend(csr_forest_trees[i][element])
            idx += len(csr_forest_trees[i][element])
        arr_idx[num_of_trees] = idx
    
    consolidate_csr(node_list_total,node_list_idx,csr_forest_trees,1)
    consolidate_csr(edge_list_total,edge_list_idx,csr_forest_trees,2)
    consolidate_csr(node_is_leaf_total,node_is_leaf_idx,csr_forest_trees,3)
    consolidate_csr(node_features_total,node_features_idx,csr_forest_trees,4)
    consolidate_csr(node_values_total,node_values_idx,csr_forest_trees,5)
    
    #node_list_idx      
    #edge_list_idx      
    #node_is_leaf_idx   
    #node_features_idx  
    #node_values_idx    
    #node_list_total 
    #edge_list_total 
    #node_is_leaf_total
    #node_features_total
    #node_values_total
    
    treename = "td" + str(_max_depth) 
    with open( treename + "_csr.txt",'w') as f:
        write_array( node_list_idx      , "node_list_idx"       ,f)  
        write_array( edge_list_idx      , "edge_list_idx"       ,f) 
        write_array( node_is_leaf_idx   , "node_is_leaf_idx"    ,f) 
        write_array( node_features_idx  , "node_features_idx"   ,f) 
        write_array( node_values_idx    , "node_values_idx"     ,f) 
        write_array( node_list_total    , "node_list_total"     ,f)  
        write_array( edge_list_total    , "edge_list_total"     ,f)  
        write_array( node_is_leaf_total , "node_is_leaf_total"  ,f) 
        write_array( node_features_total, "node_features_total" ,f)  
        write_array( node_values_total  , "node_values_total"   ,f) 
    

    ## SAVING LISTS
    lists_concat = [node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total]
    save_objects(lists_concat, "td"+str(_max_depth))

    ### SAVING ENTIRE CSR_FOREST_TREES
    save_objects(csr_forest_trees, "td_forest"+str(_max_depth))