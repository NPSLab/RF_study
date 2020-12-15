#define tra_sub_node(curr_node)   \
                    unsigned feature_id = subtree_node_list[(curr_node)*3]; \
                    float node_value    = subtree_node_list[(curr_node)*3+1]; \
                    unsigned is_tree_leaf    = subtree_node_list[(curr_node)*3+2]; \
                    if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); return_from_curr_tree = true; goto SUBTREE_END; }\
                    bool not_subtree_bottom = (curr_node) < subtree_leaf_idx_boundry; \
                    bool go_left = row[feature_id] <= node_value; \
                    if (not_subtree_bottom){ \
                        if (go_left){ \
                            tra_sub_node((curr_node)*2+1) \
                        }\
                        else{\
                            tra_sub_node((curr_node)*2+2) \
                        }\
                    } else{\
                        unsigned leaf_idx = (curr_node) - subtree_leaf_idx_boundry;\
                        if (go_left){\
                            curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx];\
                        }else{\
                            curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx+1];\
                        }\
                        goto SUBTREE_END;\
                    }
                


