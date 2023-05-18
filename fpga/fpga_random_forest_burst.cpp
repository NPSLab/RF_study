// #define MAX_SUBDEPTH 15
extern "C" {
#include "assert.h"
// #include <stdio.h>
#define MAX_NODES ((1 << MAX_SUBDEPTH)-1)
#define MAX_FEATURES 1024
const int c_size = MAX_NODES;
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

  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  float    *  queries                     ,
  unsigned * results                  
){

    for (int tid = 0; tid < num_of_queries; tid++){
      float queries_buf[MAX_FEATURES];
      // if(tid == 0) printf("New Query tid=%d\n", tid);
      //fetch a new query
      unsigned result_temp = 0;
      unsigned row_offset = tid*num_of_features;
      for(int i = 0; i < num_of_features; i++){
        queries_buf[i] = queries[row_offset + i];
      }
      // if(tid == 0) printf("Grabbed Query Info tid=%d\n", tid);
         TREE_ITERATION:
         for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
            //go over trees
            //tree_num=0;
            unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
            unsigned subtree_offset = 0;
            //unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;
        
            unsigned curr_subtree_idx = 0 ; 

            bool return_from_tree = false;
            bool return_from_subtree = false;
            //iterate over subtree 
            // if(tid == 0 && tree_num == 0) printf("Subtree Iteration tid=%d\n", tid);
            // if(tid == 0 && tree_num == 0) printf("Tree Num=%d\n", tree_num);
            SUBTREE_ITERATION:
            while (!return_from_tree){
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
                const unsigned subtree_batch_boundary = subtree_leaf_idx_boundry * 2 + 2;
        
                //fetch subtree_idx_to_other_subtree
                const unsigned subtree_idx_to_subtree_offset = g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

                // Burst access data structures
              unsigned short nodes_is_leaf_feature_id_buf[MAX_NODES]; // 2.097MB
              // #pragma HLS bind_storage variable=nodes_is_leaf_feature_id_buf type=RAM_1P impl=bram
              float nodes_value_buf[MAX_NODES]; // 4.194MB
              #pragma HLS bind_storage variable=nodes_value_buf type=RAM_1P impl=uram

              unsigned boundary = subtree_leaf_idx_boundry * 2 + 1;

              assert (boundary > 0);
              BURST_MEMORY_READ_VALUE:
              for(int i = 0; i < boundary; i++){
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
                nodes_value_buf[i] = nodes_value[subtree_node_value_list_offset + i];
              }
              BURST_MEMORY_READ_FEATURELEAF_ID:
              for(int i = 0; i < boundary; i++){
                #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
                nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + i];
              }

                //start from node 0
                unsigned curr_node = 0;

                //iterate over nodes in a subtree
                // if(tid == 0 && tree_num == 0) printf("Traverse Subnodes tid=%d\n", tid);
                SUBTREE_TRAVERSAL:
                while (!return_from_subtree){
                  //Mod
                    unsigned short feature_id = nodes_is_leaf_feature_id_buf[curr_node] >> 1;                    
                    float node_value    = nodes_value_buf[curr_node];
                    bool go_left = queries_buf[feature_id] <= node_value;
                    bool is_tree_leaf    = nodes_is_leaf_feature_id_buf[curr_node] & 0x1;
                    // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                    if (is_tree_leaf==1){
                      // if(tid == 0 && tree_num == 0) printf("Found Leaf!!! tid=%d, curr_node = %d\n", tid, curr_node);
                      result_temp += node_value;
                      return_from_subtree = true;
                      return_from_tree = true;
                    } //goto SUBTREE_END;
                    else{
                      // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                      bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                      //Mod
                      // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                      if (not_subtree_bottom){
                        #ifndef __SYNTHESIS__
                          // if(tid == 0 && tree_num == 0) printf("Inside! curr_node = %d, boundary = %d, query = %f, node_value = %f\n", curr_node, subtree_leaf_idx_boundry, queries[row_offset + feature_id], node_value);
                        #endif
                          // go to left child in subtree
                          if (go_left)
                              curr_node = curr_node*2 + 1;
                          // go to right child in subtree
                          else
                              curr_node = curr_node*2 + 2;
                      // if reach bottom of subtree, then we need to go to another subtree
                      } else{
                          unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                          #ifndef __SYNTHESIS__
                            // if(tid == 0 && tree_num == 0) printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                          #endif
                          if (go_left)
                              subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx;
                          else
                              subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx+1;
                          return_from_subtree = true;
                          //stop the iterating of the current subtree, jump to the outer loop
                      }
                    }
                    // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                    // not_subtree_bottom[curr_node] = curr_node < subtree_leaf_idx_boundry;
                }
            }
          }
      results[tid] = result_temp > num_of_trees/2;
    }
}
}
