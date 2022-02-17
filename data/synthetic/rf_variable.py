# from data.test_tree_depth.tree_depth_testing import DATASET_NAME, MAX_DEPTH, MAX_ESTIMATORS
import sklearn, numpy as np
import os
import math
import time
from numba import cuda
from cuml import ForestInference
# test classification dataset
from sklearn.datasets import make_classification
# make predictions using random forest for classification
from sklearn.ensemble import RandomForestClassifier
# split data into train and test
from sklearn.model_selection import train_test_split

from utilities import save_objects
from utilities import load_objects
from utilities import loadcsr_from_txt
from sklearn.cluster import KMeans

CUML_ITERATIONS = 25
MAX_STD = 20

#configure the dataset
_n_samples=581012*2
_n_features=54
_n_redundant=5
_n_informative= _n_features - _n_redundant 
_random_state=3

# define dataset
X_all, y_all = make_classification(n_samples=_n_samples, n_features=_n_features, n_informative=_n_informative, n_redundant=_n_redundant, random_state=_random_state)

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.5)

#configure the dataset
MODEL_PATH = r'/home/mkshah5/susy_trained//'
DATASET_NAME = "SYNTHETIC"
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


#loop over different pairs of #of estimators and max_depth
#configure the forest
#_n_estimators = 50 
#MAX_DEPTH = 60 
#_subtree_depth =  TO BE CONFIGURED IN BELOW 
NE_RANGE = [100]
TD_RANGE = [15, 25, 35]
PL_RANGE = [0.3, 0.5, 0.7, 0.9]
configs = []

for ne in NE_RANGE:
    for td in TD_RANGE:
        for pl in PL_RANGE:
            configs.append([ne, td, pl])

# configs = [[100, 10, 0.3],[100,20, 0.3],[100,30, 0.3], [100, 15, 0.6],[100,25, 0.6],[100,30, 0.6]]

for conf in configs:
    _n_estimators = conf[0]
    _max_depth = conf[1]
    max_percent_leaf = conf[2]
    
    # define the model
    model = RandomForestClassifier(n_estimators= _n_estimators, max_depth = _max_depth)
    
    # fit the model on the whole dataset
    print("Loading model")
    # model = load_objects(MODEL_PATH+"MODEL"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators))[0]
    model.fit(X, y)

    results_file = open(DATASET_NAME+"_cuml_results.txt", 'a')

    results_file.write("pl: "+ str(max_percent_leaf) +"td: " +str(_max_depth)+" ne: "+str(_n_estimators)+ "\n")
    # model = RandomForestClassifier(n_estimators= NUM_ESTIMATORS[j], max_depth = DEPTHS[j])
    # model.fit(X, y)
    # preds = model.predict(X_test)
    # score = sklearn.metrics.accuracy_score(y_test, preds)
    # results_file.write("Expected Score: "+str(score) +"\n")
    X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
    
    for i in range(0,CUML_ITERATIONS):
        fm = ForestInference.load_from_sklearn(model, output_class=True)
        start_time = time.time()
        fil_preds_gpu = fm.predict(X_gpu)
        end_time = time.time() - start_time
        accuracy_score = sklearn.metrics.accuracy_score(y_test, np.asarray(fil_preds_gpu))
        results_file.write("Average Accuracy: "+str(accuracy_score)+", Time to complete: "+str(end_time)+"\n")
    results_file.close()
    print("Ran CUML")
    
    #generate strings for each feature name used in dot file, feature[i] is str(i)
    feature_list = [str(i) for i in range(0,model.n_features_)]
    
    csr_forest_trees = []
    unsorted_csr_trees = []
    
    print("Start transforming the tree")

    clustering_input = np.zeros((_n_estimators, model.n_features_))
    
    for idx in range(0,_n_estimators):
        # print("Transforming tree:{0}".format(idx))
        curr_tree = model.estimators_[idx].tree_
        dt = model.estimators_[idx]
        # print("num features seen: "+str(dt.n_features_))
        importances = dt.feature_importances_
        for im_ind in range(len(importances)):
            if abs(importances[im_ind]) <= 1e-07:
                importances[im_ind] = 0
            else:
                importances[im_ind] = 1
        # print(dt.feature_importances_)
        clustering_input[idx, :] = importances
        num_of_nodes = curr_tree.node_count 
    
        # print("Start building connection matrix")
        
        # print("Done building connection matrix")
        #CSR format for a tree
        # print("Start building CSR format")
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
        
        # print("Done building CSR tree")
    
        #NOW These arrays are ready for the current tree/estimator X
        #num_of_nodes
        #node_list
        #edge_list
        #node_is_leaf
        #node_features
        #node_values
    
        csr_decision_tree = [num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values]
        unsorted_csr_trees.append(csr_decision_tree)
        # csr_forest_trees.append(csr_decision_tree)
        # print("Add current CSR tree into forest")
    
    kmeans = KMeans(n_clusters = int(math.sqrt(_n_estimators)), random_state=0).fit(clustering_input)
    labels=kmeans.labels_
    # print(labels)
    for cluster in range(int(math.sqrt(_n_estimators))):
        for lbl_idx in range(_n_estimators):
            if int(labels[lbl_idx]) == cluster:
                csr_forest_trees.append(unsorted_csr_trees[lbl_idx])

    print("\n\n Now we are trying to write CSR trees layouts")
    num_of_trees = len(csr_forest_trees) 
    # print("num trees: "+str(num_of_trees))
    
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
    
    treename = "clustered_variable_"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators) 
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
    

    ### SAVING LISTS
    # lists_concat = [node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total]
    # save_objects(lists_concat, "td"+str(_max_depth))

    ### SAVING ENTIRE CSR_FOREST_TREES
    # save_objects(csr_forest_trees, "td_forest"+str(_max_depth))

    ### LOADING LISTS
    # node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total = load_objects("td"+str(_max_depth))

    ### LOADING ENTIRE CSR_FOREST_TREES
    # csr_forest_trees = load_objects("td_forest"+str(_max_depth))
    #Now generate hier trees of different depth

    ### LOADING LISTS FROM TEXT FILE
    # list_mapping = loadcsr_from_txt("td"+str(_max_depth)+"_csr.txt")
    # for listname in list_mapping.keys():
    #     locals()[listname] = list_mapping[listname]
    
    
    
    for _subtree_depth in range(9,10): 
        
        forest_trees = []
        #_subtree_depth = 2 
        
        num_of_trees=len(csr_forest_trees)
        for i in range(num_of_trees):
        
            csr_decision_tree = csr_forest_trees[i]
        #    csr_decision_tree = [num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values]
            num_of_nodes  = csr_decision_tree[0] 
            node_list     = csr_decision_tree[1] 
            edge_list     = csr_decision_tree[2] 
            node_is_leaf  = csr_decision_tree[3] 
            node_features = csr_decision_tree[4] 
            node_values   = csr_decision_tree[5] 
        
            class SubTree:
                def __init__(self, subtree_max_depth , num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values, pending_subtrees_to_build,latest_tree_num ):
                    #subtree structures
        #MODIFY
                    self.subtree_max_depth = subtree_max_depth
                    #32 is ok for depth until 5
                    self.subtree_node_list = np.zeros(3*int(math.pow(2.0, float(subtree_max_depth+1)))) # has 3 attributes per node, feature_id node_value is_leaf 
                    self.subtree_size = 0
                    self.subtree_leaf_idx_boundry = 0
                    self.subtree_idx_to_other_subtree = []
            
                    #data structures of the original tree
                    self.num_of_nodes    =  num_of_nodes  
                    self.node_list       =  node_list     
                    self.edge_list       =  edge_list     
                    self.node_is_leaf    =  node_is_leaf  
                    self.node_features   =  node_features  
                    self.node_values     =  node_values   
            
                    #meta data used to help building subtrees
                    # self.depth_map       = [0,1,3,7,15] # local list, stores the first node's index for depth 0,1,2,3
                    self.depth_map = [0]
                    for i in range(0, subtree_max_depth):
                        res = self.depth_map[i] + int(math.pow(2.0, float(i)))
                        self.depth_map.append(res)
                    self.pending_subtrees_to_build = pending_subtrees_to_build   #external list, stores pending subtrees to be built 
                    self.latest_tree_num = latest_tree_num # external list, stores the last tree number generated by current subtree
                    self.pending_subtrees_idx = [] #local list, stores new subtree ids generated by current subtree, used to generate subtree_idx_to_other_subtree array
            
                def build_subtree(self,node_id,subtree_node_id,depth):
                    if int(node_id) == 0:
                        level = 0
                        curr_node = 0
                        nodes_to_check = 1
                        depths = [0]
                        st_var_max = 0
                        percent_leaf = 0.0
                        leaves = 0
                        prev_curr_node = 0
                        while curr_node < nodes_to_check:
                            has_leaf = False
                            leaves = 0
                            if ((nodes_to_check+node_id-1) >= len(self.node_is_leaf)) or (level >= MAX_STD):
                                break
                            for i in range(int(curr_node+node_id), int(nodes_to_check+node_id - 1)):
                                
                                if self.node_is_leaf[i] == 1:
                                    leaves+=1
                                    has_leaf = True
                            percent_leaf = float(leaves)/float(nodes_to_check-prev_curr_node)
                            if percent_leaf >= max_percent_leaf:
                                break
                            else:
                                level+=1
                                depths.append(int(nodes_to_check))
                                prev_curr_node = curr_node
                                curr_node = nodes_to_check
                                nodes_to_check = nodes_to_check + int(math.pow(2.0, float(level)))
                        
                        self.subtree_max_depth = level
                        self.depth_map = depths
                        self.subtree_node_list = np.zeros(3*int(math.pow(2.0, float(level+1))))
                        # if node_id == 0:
                        print(" max_depth: "+str(self.subtree_max_depth)+ " percent leaf: "+str(percent_leaf))
                    # if node_id == 0:
                    #     self.depth_map       = [0,1,3,7,15,31,63,127,255,511,1023, 2047, 4095, 8191, 16383]
                    #     self.subtree_max_depth = 14
                    #     self.subtree_node_list = np.zeros(3*32768)

                    #store the deepest depth/leaf_idx_boundry that has been reached during building current subtree
                    leaf_idx_boundry = self.depth_map[depth]
                    if leaf_idx_boundry > self.subtree_leaf_idx_boundry:
                        self.subtree_leaf_idx_boundry = leaf_idx_boundry
                    
                    outstr = "build subtree node {subtree_node_id}, original node id is {node_id}".format(subtree_node_id = subtree_node_id, node_id = node_id)
                    #print(outstr)
            
                    self.subtree_node_list[3*subtree_node_id] = self.node_features[node_id]
                    self.subtree_node_list[3*subtree_node_id+1] = self.node_values[node_id]
                    self.subtree_node_list[3*subtree_node_id+2] = self.node_is_leaf[node_id]
           
                    #subtree_size keeps the node with largest local index inside the subtree, tree node starts from 0, so we + 1 here 
                    if self.subtree_size < subtree_node_id+1:
                        self.subtree_size = subtree_node_id+1
           
                    
                    if self.node_is_leaf[node_id] != 1:
                        edge_idx = self.node_list[node_id]
                        left_node_id = self.edge_list[edge_idx] 
                        right_node_id = self.edge_list[edge_idx+1]
        #DEPTH
                        if depth < self.subtree_max_depth :
                            self.build_subtree(left_node_id,2*subtree_node_id+1,depth+1)
                            self.build_subtree(right_node_id,2*subtree_node_id+2,depth+1)
                        else:
            
                            #print("need to build new tree at original node {node_id}".format(node_id=left_node_id))
                            self.latest_tree_num[0]+=1
                            self.pending_subtrees_to_build.append([left_node_id,self.latest_tree_num[0]])
            
                            #print("need to build new tree at original node {node_id}".format(node_id=right_node_id))
                            self.latest_tree_num[0]+=1
                            self.pending_subtrees_to_build.append([right_node_id,self.latest_tree_num[0]])
            
                            self.pending_subtrees_idx.append([subtree_node_id,self.latest_tree_num[0]-1,self.latest_tree_num[0]])
                            return
                    else:
                        #print("Meet leaf node {node_id}".format(node_id=node_id))
                        pass
                def build_subtree_idx_to_other_subtree(self):
                    if len(self.pending_subtrees_idx) == 0:
                        return
                    last_node_to_subtree = self.pending_subtrees_idx[-1][0]
                    #calculate how many nodes are in last level
                    self.num_last_level_node = last_node_to_subtree - self.subtree_leaf_idx_boundry + 1
                    self.subtree_idx_to_other_subtree = [-1]*(2*self.num_last_level_node)
                    for record in self.pending_subtrees_idx:
                        node_to_other_subtree = record[0]
                        idx = node_to_other_subtree - self.subtree_leaf_idx_boundry
                        self.subtree_idx_to_other_subtree[idx*2+0] = record[1]
                        self.subtree_idx_to_other_subtree[idx*2+1] = record[2]
           
            #build current hier_tree
            decision_tree={}
            pending_subtrees_to_build = [[0,0]]
            latest_tree_num = [0]
        
            while len(pending_subtrees_to_build) > 0:
            
                ori_node_id = pending_subtrees_to_build[0][0]
                sub_tree_num = pending_subtrees_to_build[0][1]
                
                pending_subtrees_to_build.pop(0)
                
                subtree = SubTree(_subtree_depth, num_of_nodes ,node_list ,edge_list ,node_is_leaf ,node_features ,node_values , pending_subtrees_to_build, latest_tree_num)
                
                subtree.build_subtree(node_id=ori_node_id,subtree_node_id=0,depth=0)
                subtree.build_subtree_idx_to_other_subtree()
                
                decision_tree[sub_tree_num]=subtree
            
            # print("Done transforming tree")
            #Now decision_tree contains all subtrees, start traversal with subtree0
            forest_trees.append(decision_tree)
            # print("Add tree_hier into forest")
        
        #  now we have forest_trees of decision_tree, each decision tree comprise subtrees that have the following attributes
        #            subtree.subtree_node_list = np.zeros(3*15) # has 3 attributes per node, feature_id node_value is_leaf 
        #            subtree.subtree_size = 0
        #            subtree.subtree_leaf_idx_boundry = 0
        #            subtree.subtree_idx_to_other_subtree = []
        #            decision_tree[sub_tree_num]=subtree
        #            forest_trees.append(decision_tree)
        
        num_of_trees = len(forest_trees) 
        
        sum_subtree_nums = []
        total_num_node = 0
        
        for t in forest_trees :
            sum_subtree_nums.append(len(t))
            for st_num in t:
                st = t[st_num]
                total_num_node += st.subtree_size
        
        prefix_sum_subtree_nums = [sum(sum_subtree_nums[:i+1]) for i in range(len(sum_subtree_nums))] 
        prefix_sum_subtree_nums.insert(0,0)
        
        #Now generate nodes and g_subtree_nodes_offset, verify t_subtree_nodes_start
        nodes = []
        g_subtree_nodes_offset = []
        nodes_idx = 0
        
        idx_to_subtree = []
        g_subtree_idx_to_subtree_offset = []
        idx_to_subtree_idx = 0
        
        leaf_idx_boundry = []
        
        for t in forest_trees:
            #update to verify t_subtree_nodes_offset
            #t_subtree_nodes_start.append(nodes_idx)
        
            for st_num in t:
                st = t[st_num]
                #update g offset array
                g_subtree_nodes_offset.append(nodes_idx)
                g_subtree_idx_to_subtree_offset.append(idx_to_subtree_idx)
        
                #update nodes, notice that 3 value tuple is extended to the end
                nodes.extend(st.subtree_node_list[:3*st.subtree_size])
                #update idx_to_subtree, notice that every leaf node reserves two positions
                idx_to_subtree.extend(st.subtree_idx_to_other_subtree)
                #update leaf_idx_boundry
                leaf_idx_boundry.append(st.subtree_leaf_idx_boundry)
        
                #update nodes_idx, based on 3-value tuple, not number of values
                nodes_idx += st.subtree_size 
                idx_to_subtree_idx += len(st.subtree_idx_to_other_subtree)/2 
        
        #now we have these layout to use on GPU
        #       num_of_trees
        #       prefix_sum_subtree_nums
        #       
        #       nodes
        #       idx_to_subtree
        #       leaf_idx_boundry
        #       
        #       g_subtree_nodes_offset
        #       g_subtree_idx_to_subtree_offset
        print("\n\n Now we are trying to write treefile_hier layouts")
        
        
        treename = "clustered_variable_"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators) + "_sd" + str(_subtree_depth) + "_pl"+str(max_percent_leaf)
        with open(treename + "_hier.txt",'w') as f:
            f.write("num_of_trees\n")
            f.write("{0}, \n".format(num_of_trees))
            
            f.write("prefix_sum_subtree_nums\n")
            f.write("{0},\n".format(len(prefix_sum_subtree_nums)))
            for val in prefix_sum_subtree_nums: 
                f.write("{0}, ".format(val))
            f.write("\n")
            
            f.write("nodes\n")
            f.write("{0},\n".format(len(nodes)))
            for val in nodes:
                f.write("{0}, ".format(val))
            f.write("\n")
            
            f.write("idx_to_subtree\n")
            f.write("{0},\n".format(len(idx_to_subtree)))
            for val in idx_to_subtree:
                f.write("{0}, ".format(val))
            f.write("\n")
            
            f.write("leaf_idx_boundry\n")
            f.write("{0},\n".format(len(leaf_idx_boundry)))
            for val in leaf_idx_boundry:
                f.write("{0}, ".format(val))
            f.write("\n")
            
            f.write("g_subtree_nodes_offset\n")
            f.write("{0},\n".format(len(g_subtree_nodes_offset)))
            for val in g_subtree_nodes_offset:
                f.write("{0}, ".format(val))
            f.write("\n")
            
            f.write("g_subtree_idx_to_subtree_offset\n")
            f.write("{0},\n".format(len(g_subtree_idx_to_subtree_offset)))
            for val in g_subtree_idx_to_subtree_offset:
                f.write("{0}, ".format(val))
            f.write("\n")

#_max_depth = 45
#MAX_ESTIMATORS = 100

#for _n_estimators in range(10,MAX_ESTIMATORS+1, 10):    
#    
#    # define the model
#    # model = RandomForestClassifier(n_estimators= _n_estimators, max_depth = _max_depth)
#    
#    # fit the model on the whole dataset
#    print("Loading model")
#    model = load_objects(MODEL_PATH+"MODEL"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators))[0]
#    # model.fit(X, y)
#    
#    #generate strings for each feature name used in dot file, feature[i] is str(i)
#    feature_list = [str(i) for i in range(0,model.n_features_)]
#    
#    csr_forest_trees = []
#    
#    unsorted_csr_trees = []
#    
#    print("Start transforming the tree")
#
#    clustering_input = np.zeros((_n_estimators, model.n_features_))
#    
#    for idx in range(0,_n_estimators):
#        print("Transforming tree:{0}".format(idx))
#        curr_tree = model.estimators_[idx].tree_
#        dt = model.estimators_[idx]
#        # print("num features seen: "+str(dt.n_features_))
#        importances = dt.feature_importances_
#        for im_ind in range(len(importances)):
#            if abs(importances[im_ind]) <= 1e-07:
#                importances[im_ind] = 0
#            else:
#                importances[im_ind] = 1
#        # print(dt.feature_importances_)
#        clustering_input[idx, :] = importances
#        num_of_nodes = curr_tree.node_count 
#    
#        print("Start building connection matrix")
#        
#        print("Done building connection matrix")
#        #CSR format for a tree
#        print("Start building CSR format")
#        #index of each node into edge list 
#        node_list = np.zeros(num_of_nodes+1,dtype='i') 
#        #reserve 2*N spaces, in fact, will be less than 2N, stores all edges
#        edge_list = np.zeros(2*num_of_nodes,dtype='i') 
#        #indicate of node is leaf, redundant, can be deducted from node_list, if node_list[i+1] = node_list[i], then node i is leaf node
#        node_is_leaf = np.zeros(num_of_nodes,dtype='i') 
#        curr_idx = 0
#    
#        for i in range(num_of_nodes):
#            node_list[i] = curr_idx
#            found_one = 0
#    
#            if curr_tree.children_left[i] != -1:
#                edge_list[curr_idx] = curr_tree.children_left[i]
#                curr_idx+=1
#                found_one = 1
#    
#            if curr_tree.children_right[i] != -1:
#                edge_list[curr_idx] = curr_tree.children_right[i]
#                curr_idx+=1
#                found_one = 1
#    
#            if found_one == 0:
#                node_is_leaf[i]=1
#        
#        node_list[num_of_nodes] = curr_idx
#        edge_list.resize(curr_idx)
#        
#        # read feature number and value used in each node
#        node_values = np.zeros(num_of_nodes) 
#        
#        node_features = np.zeros(num_of_nodes, dtype='i') 
#        
#        for i in range(num_of_nodes):
#            #mata_string = graph.get_node(str(i))[0].get_label()
#            if node_is_leaf[i]:
#                if model.estimators_[idx].tree_.value[i][0][0] >= model.estimators_[idx].tree_.value[i][0][1] : 
#                    node_values[i]=0
#                else:
#                    node_values[i]=1
#            else:
#                node_features[i] = model.estimators_[idx].tree_.feature[i] 
#                node_values[i] = model.estimators_[idx].tree_.threshold[i] 
#        
#        print("Done building CSR tree")
#    
#        #NOW These arrays are ready for the current tree/estimator X
#        #num_of_nodes
#        #node_list
#        #edge_list
#        #node_is_leaf
#        #node_features
#        #node_values
#    
#        csr_decision_tree = [num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values]
#        unsorted_csr_trees.append(csr_decision_tree)
#        print("Add current CSR tree into forest")
#    
#    kmeans = KMeans(n_clusters = int(math.sqrt(_n_estimators)), random_state=0).fit(clustering_input)
#    labels=kmeans.labels_
#    # print(labels)
#    for cluster in range(int(math.sqrt(_n_estimators))):
#        for lbl_idx in range(_n_estimators):
#            if int(labels[lbl_idx]) == cluster:
#                csr_forest_trees.append(unsorted_csr_trees[lbl_idx])
#    
#    print("\n\n Now we are trying to write CSR trees layouts")
#    num_of_trees = len(csr_forest_trees) 
#    
#    if num_of_trees != _n_estimators:
#        print("error")
#    
#    node_list_idx      = np.zeros(num_of_trees+1,dtype='i')
#    edge_list_idx      = np.zeros(num_of_trees+1,dtype='i')
#    node_is_leaf_idx   = np.zeros(num_of_trees+1,dtype='i')
#    node_features_idx  = np.zeros(num_of_trees+1,dtype='i')
#    node_values_idx    = np.zeros(num_of_trees+1,dtype='i')
#    
#    node_list_total = []
#    edge_list_total = []
#    node_is_leaf_total =[]
#    node_features_total = []
#    node_values_total = []
#    #consolidate trees into a signle array
#    def consolidate_csr(arr,arr_idx,csr_forest_trees,element):
#        num_of_trees=len(csr_forest_trees)
#        idx = 0
#        for i in range(num_of_trees):
#            arr_idx[i] = idx 
#            arr.extend(csr_forest_trees[i][element])
#            idx += len(csr_forest_trees[i][element])
#        arr_idx[num_of_trees] = idx
#    
#    consolidate_csr(node_list_total,node_list_idx,csr_forest_trees,1)
#    consolidate_csr(edge_list_total,edge_list_idx,csr_forest_trees,2)
#    consolidate_csr(node_is_leaf_total,node_is_leaf_idx,csr_forest_trees,3)
#    consolidate_csr(node_features_total,node_features_idx,csr_forest_trees,4)
#    consolidate_csr(node_values_total,node_values_idx,csr_forest_trees,5)
#    
#    #node_list_idx      
#    #edge_list_idx      
#    #node_is_leaf_idx   
#    #node_features_idx  
#    #node_values_idx    
#    #node_list_total 
#    #edge_list_total 
#    #node_is_leaf_total
#    #node_features_total
#    #node_values_total
#    
#    treename = "clustered"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators)
#    with open( treename + "_csr.txt",'w') as f:
#        write_array( node_list_idx      , "node_list_idx"       ,f)  
#        write_array( edge_list_idx      , "edge_list_idx"       ,f) 
#        write_array( node_is_leaf_idx   , "node_is_leaf_idx"    ,f) 
#        write_array( node_features_idx  , "node_features_idx"   ,f) 
#        write_array( node_values_idx    , "node_values_idx"     ,f) 
#        write_array( node_list_total    , "node_list_total"     ,f)  
#        write_array( edge_list_total    , "edge_list_total"     ,f)  
#        write_array( node_is_leaf_total , "node_is_leaf_total"  ,f) 
#        write_array( node_features_total, "node_features_total" ,f)  
#        write_array( node_values_total  , "node_values_total"   ,f) 
#    
#
#    ### SAVING LISTS
#    # lists_concat = [node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total]
#    # save_objects(lists_concat, "td"+str(_max_depth))
#
#    ### SAVING ENTIRE CSR_FOREST_TREES
#    # save_objects(csr_forest_trees, "td_forest"+str(_max_depth))
#
#    ### LOADING LISTS
#    # node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total = load_objects("td"+str(_max_depth))
#
#    ### LOADING ENTIRE CSR_FOREST_TREES
#    # csr_forest_trees = load_objects("td_forest"+str(_max_depth))
#    #Now generate hier trees of different depth
#
#    ### LOADING LISTS FROM TEXT FILE
#    # list_mapping = loadcsr_from_txt("td"+str(_max_depth)+"_csr.txt")
#    # for listname in list_mapping.keys():
#    #     locals()[listname] = list_mapping[listname]
#    
#    
#    
#    for _subtree_depth in range(2,5): 
#        
#        forest_trees = []
#        #_subtree_depth = 2 
#        
#        num_of_trees=len(csr_forest_trees)
#        for i in range(num_of_trees):
#        
#            csr_decision_tree = csr_forest_trees[i]
#        #    csr_decision_tree = [num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values]
#            num_of_nodes  = csr_decision_tree[0] 
#            node_list     = csr_decision_tree[1] 
#            edge_list     = csr_decision_tree[2] 
#            node_is_leaf  = csr_decision_tree[3] 
#            node_features = csr_decision_tree[4] 
#            node_values   = csr_decision_tree[5] 
#        
#            class SubTree:
#                def __init__(self, subtree_max_depth , num_of_nodes, node_list, edge_list, node_is_leaf, node_features, node_values, pending_subtrees_to_build,latest_tree_num ):
#                    #subtree structures
#        #MODIFY
#                    self.subtree_max_depth = subtree_max_depth
#                    #32 is ok for depth until 5
#                    self.subtree_node_list = np.zeros(3*32) # has 3 attributes per node, feature_id node_value is_leaf 
#                    self.subtree_size = 0
#                    self.subtree_leaf_idx_boundry = 0
#                    self.subtree_idx_to_other_subtree = []
#            
#                    #data structures of the original tree
#                    self.num_of_nodes    =  num_of_nodes  
#                    self.node_list       =  node_list     
#                    self.edge_list       =  edge_list     
#                    self.node_is_leaf    =  node_is_leaf  
#                    self.node_features   =  node_features  
#                    self.node_values     =  node_values   
#            
#                    #meta data used to help building subtrees
#                    self.depth_map       = [0,1,3,7,15] # local list, stores the first node's index for depth 0,1,2,3
#                    self.pending_subtrees_to_build = pending_subtrees_to_build   #external list, stores pending subtrees to be built 
#                    self.latest_tree_num = latest_tree_num # external list, stores the last tree number generated by current subtree
#                    self.pending_subtrees_idx = [] #local list, stores new subtree ids generated by current subtree, used to generate subtree_idx_to_other_subtree array
#            
#                def build_subtree(self,node_id,subtree_node_id,depth):
#        
#                    #store the deepest depth/leaf_idx_boundry that has been reached during building current subtree
#                    leaf_idx_boundry = self.depth_map[depth]
#                    if leaf_idx_boundry > self.subtree_leaf_idx_boundry:
#                        self.subtree_leaf_idx_boundry = leaf_idx_boundry
#                    
#                    outstr = "build subtree node {subtree_node_id}, original node id is {node_id}".format(subtree_node_id = subtree_node_id, node_id = node_id)
#                    #print(outstr)
#            
#                    self.subtree_node_list[3*subtree_node_id] = self.node_features[node_id]
#                    self.subtree_node_list[3*subtree_node_id+1] = self.node_values[node_id]
#                    self.subtree_node_list[3*subtree_node_id+2] = self.node_is_leaf[node_id]
#           
#                    #subtree_size keeps the node with largest local index inside the subtree, tree node starts from 0, so we + 1 here 
#                    if self.subtree_size < subtree_node_id+1:
#                        self.subtree_size = subtree_node_id+1
#           
#                    
#                    if self.node_is_leaf[node_id] != 1:
#                        edge_idx = self.node_list[node_id]
#                        left_node_id = self.edge_list[edge_idx] 
#                        right_node_id = self.edge_list[edge_idx+1]
#        #DEPTH
#                        if depth < self.subtree_max_depth :
#                            self.build_subtree(left_node_id,2*subtree_node_id+1,depth+1)
#                            self.build_subtree(right_node_id,2*subtree_node_id+2,depth+1)
#                        else:
#            
#                            #print("need to build new tree at original node {node_id}".format(node_id=left_node_id))
#                            self.latest_tree_num[0]+=1
#                            self.pending_subtrees_to_build.append([left_node_id,self.latest_tree_num[0]])
#            
#                            #print("need to build new tree at original node {node_id}".format(node_id=right_node_id))
#                            self.latest_tree_num[0]+=1
#                            self.pending_subtrees_to_build.append([right_node_id,self.latest_tree_num[0]])
#            
#                            self.pending_subtrees_idx.append([subtree_node_id,self.latest_tree_num[0]-1,self.latest_tree_num[0]])
#                            return
#                    else:
#                        #print("Meet leaf node {node_id}".format(node_id=node_id))
#                        pass
#                def build_subtree_idx_to_other_subtree(self):
#                    if len(self.pending_subtrees_idx) == 0:
#                        return
#                    last_node_to_subtree = self.pending_subtrees_idx[-1][0]
#                    #calculate how many nodes are in last level
#                    self.num_last_level_node = last_node_to_subtree - self.subtree_leaf_idx_boundry + 1
#                    self.subtree_idx_to_other_subtree = [-1]*(2*self.num_last_level_node)
#                    for record in self.pending_subtrees_idx:
#                        node_to_other_subtree = record[0]
#                        idx = node_to_other_subtree - self.subtree_leaf_idx_boundry
#                        self.subtree_idx_to_other_subtree[idx*2+0] = record[1]
#                        self.subtree_idx_to_other_subtree[idx*2+1] = record[2]
#           
#            #build current hier_tree
#            decision_tree={}
#            pending_subtrees_to_build = [[0,0]]
#            latest_tree_num = [0]
#        
#            while len(pending_subtrees_to_build) > 0:
#            
#                ori_node_id = pending_subtrees_to_build[0][0]
#                sub_tree_num = pending_subtrees_to_build[0][1]
#                
#                pending_subtrees_to_build.pop(0)
#                
#                subtree = SubTree(_subtree_depth, num_of_nodes ,node_list ,edge_list ,node_is_leaf ,node_features ,node_values , pending_subtrees_to_build, latest_tree_num)
#                
#                subtree.build_subtree(node_id=ori_node_id,subtree_node_id=0,depth=0)
#                subtree.build_subtree_idx_to_other_subtree()
#                
#                decision_tree[sub_tree_num]=subtree
#            
#            print("Done transforming tree")
#            #Now decision_tree contains all subtrees, start traversal with subtree0
#            forest_trees.append(decision_tree)
#            print("Add tree_hier into forest")
#        
#        #  now we have forest_trees of decision_tree, each decision tree comprise subtrees that have the following attributes
#        #            subtree.subtree_node_list = np.zeros(3*15) # has 3 attributes per node, feature_id node_value is_leaf 
#        #            subtree.subtree_size = 0
#        #            subtree.subtree_leaf_idx_boundry = 0
#        #            subtree.subtree_idx_to_other_subtree = []
#        #            decision_tree[sub_tree_num]=subtree
#        #            forest_trees.append(decision_tree)
#        
#        num_of_trees = len(forest_trees) 
#        
#        sum_subtree_nums = []
#        total_num_node = 0
#        
#        for t in forest_trees :
#            sum_subtree_nums.append(len(t))
#            for st_num in t:
#                st = t[st_num]
#                total_num_node += st.subtree_size
#        
#        prefix_sum_subtree_nums = [sum(sum_subtree_nums[:i+1]) for i in range(len(sum_subtree_nums))] 
#        prefix_sum_subtree_nums.insert(0,0)
#        
#        #Now generate nodes and g_subtree_nodes_offset, verify t_subtree_nodes_start
#        nodes = []
#        g_subtree_nodes_offset = []
#        nodes_idx = 0
#        
#        idx_to_subtree = []
#        g_subtree_idx_to_subtree_offset = []
#        idx_to_subtree_idx = 0
#        
#        leaf_idx_boundry = []
#        
#        for t in forest_trees:
#            #update to verify t_subtree_nodes_offset
#            #t_subtree_nodes_start.append(nodes_idx)
#        
#            for st_num in t:
#                st = t[st_num]
#                #update g offset array
#                g_subtree_nodes_offset.append(nodes_idx)
#                g_subtree_idx_to_subtree_offset.append(idx_to_subtree_idx)
#        
#                #update nodes, notice that 3 value tuple is extended to the end
#                nodes.extend(st.subtree_node_list[:3*st.subtree_size])
#                #update idx_to_subtree, notice that every leaf node reserves two positions
#                idx_to_subtree.extend(st.subtree_idx_to_other_subtree)
#                #update leaf_idx_boundry
#                leaf_idx_boundry.append(st.subtree_leaf_idx_boundry)
#        
#                #update nodes_idx, based on 3-value tuple, not number of values
#                nodes_idx += st.subtree_size 
#                idx_to_subtree_idx += len(st.subtree_idx_to_other_subtree)/2 
#        
#        #now we have these layout to use on GPU
#        #       num_of_trees
#        #       prefix_sum_subtree_nums
#        #       
#        #       nodes
#        #       idx_to_subtree
#        #       leaf_idx_boundry
#        #       
#        #       g_subtree_nodes_offset
#        #       g_subtree_idx_to_subtree_offset
#        print("\n\n Now we are trying to write treefile_hier layouts")
#        
#        
#        treename = "clustered"+DATASET_NAME+"_td"+str(_max_depth)+"_ne"+str(_n_estimators) + "_sd" + str(_subtree_depth)
#        with open(treename + "_hier.txt",'w') as f:
#            f.write("num_of_trees\n")
#            f.write("{0}, \n".format(num_of_trees))
#            
#            f.write("prefix_sum_subtree_nums\n")
#            f.write("{0},\n".format(len(prefix_sum_subtree_nums)))
#            for val in prefix_sum_subtree_nums: 
#                f.write("{0}, ".format(val))
#            f.write("\n")
#            
#            f.write("nodes\n")
#            f.write("{0},\n".format(len(nodes)))
#            for val in nodes:
#                f.write("{0}, ".format(val))
#            f.write("\n")
#            
#            f.write("idx_to_subtree\n")
#            f.write("{0},\n".format(len(idx_to_subtree)))
#            for val in idx_to_subtree:
#                f.write("{0}, ".format(val))
#            f.write("\n")
#            
#            f.write("leaf_idx_boundry\n")
#            f.write("{0},\n".format(len(leaf_idx_boundry)))
#            for val in leaf_idx_boundry:
#                f.write("{0}, ".format(val))
#            f.write("\n")
#            
#            f.write("g_subtree_nodes_offset\n")
#            f.write("{0},\n".format(len(g_subtree_nodes_offset)))
#            for val in g_subtree_nodes_offset:
#                f.write("{0}, ".format(val))
#            f.write("\n")
#            
#            f.write("g_subtree_idx_to_subtree_offset\n")
#            f.write("{0},\n".format(len(g_subtree_idx_to_subtree_offset)))
#            for val in g_subtree_idx_to_subtree_offset:
#                f.write("{0}, ".format(val))
#            f.write("\n")
