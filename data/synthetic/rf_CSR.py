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
# X_all, y_all = make_classification(n_samples=_n_samples, n_features=_n_features, n_informative=_n_informative, n_redundant=_n_redundant, random_state=_random_state)

# X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.5)

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

_n_estimators = 100 
_max_depth = 15 
#_subtree_depth =  TO BE CONFIGURED IN BELOW 

for _max_depth in range(7,16):   
### LOADING LISTS
    node_list_idx, edge_list_idx, node_is_leaf_idx, node_features_idx, node_values_idx, node_list_total,edge_list_total,node_is_leaf_total, node_features_total, node_values_total = load_objects("td"+str(_max_depth))

    ## LOADING ENTIRE CSR_FOREST_TREES
    csr_forest_trees = load_objects("td_forest"+str(_max_depth))
    # Now generate hier trees of different depth

    ## LOADING LISTS FROM TEXT FILE
    # list_mapping = loadcsr_from_txt("td"+str(_max_depth)+"_csr.txt")
    # for listname in list_mapping.keys():
    #     locals()[listname] = list_mapping[listname]

    for _subtree_depth in range(2,5): 
        
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
                    self.subtree_node_list = np.zeros(3*32) # has 3 attributes per node, feature_id node_value is_leaf 
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
                    self.depth_map       = [0,1,3,7,15] # local list, stores the first node's index for depth 0,1,2,3
                    self.pending_subtrees_to_build = pending_subtrees_to_build   #external list, stores pending subtrees to be built 
                    self.latest_tree_num = latest_tree_num # external list, stores the last tree number generated by current subtree
                    self.pending_subtrees_idx = [] #local list, stores new subtree ids generated by current subtree, used to generate subtree_idx_to_other_subtree array
            
                def build_subtree(self,node_id,subtree_node_id,depth):
        
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
            
            print("Done transforming tree")
            #Now decision_tree contains all subtrees, start traversal with subtree0
            forest_trees.append(decision_tree)
            print("Add tree_hier into forest")
        
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
        
        
        treename = "td" + str(_max_depth) + "sd" + str(_subtree_depth)
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