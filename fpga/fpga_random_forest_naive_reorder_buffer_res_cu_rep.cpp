// #ifdef FPGA_HIER
extern "C" {
  // #include <stdio.h>
#define MAX_FEATURES 1024
void hier_kernel(
  const unsigned num_of_trees                       ,
  unsigned *  prefix_sum_subtree_nums        ,
  /*Mod node layout*/
  float    * nodes_value                    ,
  unsigned short * nodes_is_leaf_feature_id ,

  unsigned * idx_to_subtree                 ,  
  unsigned * leaf_idx_boundry               ,
  unsigned * g_subtree_nodes_offset         ,  
  unsigned * g_subtree_idx_to_subtree_offset,  

  const unsigned start_tid                  ,
  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  float    *  queries                     ,
  unsigned * results                  
){
    for (int tid = start_tid; tid < num_of_queries; tid++){
      #pragma HLS LOOP_TRIPCOUNT min = 9000 max = 46875 avg = 25000
      // if(tid == 0) printf("New Query tid=%d\n", tid);
      //fetch a new query
      unsigned result_temp = 0;
      unsigned row_offset = tid*num_of_features;
      float queries_buf[MAX_FEATURES];
      for(int i = 0; i < num_of_features; i++){
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 54 avg = 25
        queries_buf[i] = queries[row_offset + i];
      }

         TREE_ITERATION:
         for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 100 avg = 50
            //go over trees
            //tree_num=0;
            unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
            // if(tid == 0 && tree_num == 0) printf("tree_off_set = %d\n", tree_off_set);
            //unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;

            unsigned curr_subtree_idx = 0 ;
            unsigned subtree_offset = 0;
            bool return_from_subtree = false;

            // if(tid == 0 && tree_num == 0) printf("Subtree Iteration tid=%d\n", tid);
            // if(tid == 0 && tree_num == 0) printf("Tree Num=%d\n", tree_num);

            SUBTREE_ITERATION:
            //iterate over subtree
            while (true){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 2
              // if(tid == 0 && tree_num == 0) printf("Subtree Index = %d\n", curr_subtree_idx);
                //fetch the subtree nodes
              if(return_from_subtree){
                curr_subtree_idx = idx_to_subtree[subtree_offset];
                return_from_subtree = false;
              }
              //Mod
                const unsigned subtree_node_value_list_offset
                    = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

                const unsigned subtree_node_is_leaf_feature_id_list_offset
                    = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

                //fetch subtree_leaf_idx_boundry
                const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+curr_subtree_idx];

                //fetch subtree_idx_to_other_subtree
                const unsigned subtree_idx_to_subtree_offset = g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

                //iterate over nodes in a subtree
                bool return_from_curr_tree = false;

                //start from node 0
                unsigned curr_node = 0;
                SUBTREE_TRAVERSAL:
                while (true){
                  #pragma HLS LOOP_TRIPCOUNT min = 1 max = 18 avg = 10
                  //Mod
                    unsigned short feature_id = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + curr_node] >> 1;
                    float node_value    = nodes_value[subtree_node_value_list_offset + curr_node];
                    unsigned short is_tree_leaf    = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + curr_node] & 0x1;
                    // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                    //if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); return_from_curr_tree = true; break; }
                    // if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); return_from_curr_tree = true; goto SUBTREE_END; }
                    if (is_tree_leaf==1){ result_temp += node_value; return_from_curr_tree = true; //if(tid == 499){
                        // if(tid == 0 && tree_num == 0) printf("Leaf! curr_node = %d, curr_subtree_idx = %d, tid = %d\n", curr_node, curr_subtree_idx, tid);
                      goto SUBTREE_END;
                    }
                    // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                    bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                    bool go_left = queries_buf[feature_id] <= node_value;
                    // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                    if (not_subtree_bottom){
                      // if(tid == 0 && tree_num == 0) printf("Inside! curr_node = %d, boundary = %d\n", curr_node, subtree_leaf_idx_boundry);
                        // go to left child in subtree
                        // if(tid == 0 && tree_num == 0) printf("Moving from %d to", curr_node);
                        if (go_left)
                            curr_node = curr_node*2 + 1;
                        // go to right child in subtree
                        else
                            curr_node = curr_node*2 + 2;
                        // if(tid == 0 && tree_num == 0) printf(" %d\n", curr_node);
                    // if reach bottom of subtree, then we need to go to another subtree
                    } else{
                        unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                        // if(tid == 0 && tree_num == 0) printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                        if (go_left)
                            subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx;
                        else
                            subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx+1;
                        //stop the iterating of the current subtree, jump to the outer loop
                        //break;
                        return_from_subtree = true;
                        goto SUBTREE_END;
                    }
                }
                //end subtree
                //if return from curr tree, skip all rest of subtrees, break from looping over subtrees
SUBTREE_END:
                if (return_from_curr_tree) break;
            }
          }
         unsigned threshold = num_of_trees/2;
         results[tid] = result_temp > threshold;
        }
}
}