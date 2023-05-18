// #ifdef FPGA_HIER
// #define MAX_SUBDEPTH 20
#define MAX_NODES ((1 << MAX_SUBDEPTH)-1)
__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_zero_global_work_offset))
__attribute__ ((xcl_dataflow))
void hier_kernel(
  const unsigned num_of_trees                       ,
  global unsigned *  prefix_sum_subtree_nums        ,
  /*Mod node layout*/
  global float    * nodes_value                    ,
  global unsigned short * nodes_is_leaf_feature_id ,

  global unsigned * idx_to_subtree                 ,
  global unsigned * leaf_idx_boundry               ,
  global unsigned * g_subtree_nodes_offset         ,
  global unsigned * g_subtree_idx_to_subtree_offset,

  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  global float    *  queries                     ,
  global unsigned * results
){
    // printf("Kernel start!!!\n");
    uchar query_results[375000];
    __attribute__((xcl_loop_tripcount(100, 128, 100)))
        TREE_ITERATION:
        for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
          unsigned query_curr_subtree[375000] __attribute__(xcl_resource());
          unsigned query_data_buf[375000];

          //go over trees
          //tree_num=0;
          unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];

          //iterate over subtree
          // printf("Tree Num=%d\n", tree_num);
          __attribute__((xcl_loop_tripcount(1, 524288, 10000)))
          SUBTREE_ITERATION:
          for(unsigned curr_subtree_idx = 0;
              curr_subtree_idx < prefix_sum_subtree_nums[tree_num+1] - tree_off_set;
              curr_subtree_idx++){
              //fetch the subtree nodes
            //Mod

              const unsigned subtree_node_value_list_offset
                  = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

              const unsigned subtree_node_is_leaf_feature_id_list_offset
                  = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

              //fetch subtree_leaf_idx_boundry
              const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+curr_subtree_idx];

              //fetch subtree_idx_to_other_subtree
              const unsigned subtree_idx_to_subtree_offset = g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

              // Burst access data structures
              unsigned short nodes_is_leaf_feature_id_buf[MAX_NODES];
              float nodes_value_buf[MAX_NODES];
              unsigned idx_to_subtree_buf[MAX_NODES+1];

              unsigned boundary = subtree_leaf_idx_boundry * 2 + 1;

              uchar current_depth = 1;
              unsigned boundary_temp = boundary;
              for(current_depth; boundary_temp != 1; current_depth++){
                boundary_temp = boundary_temp >> 1;
              }

              // printf("Subtree Index = %d, Boundary = %d, depth = %d\n", curr_subtree_idx, boundary, current_depth);
              // printf("Boundary = %d, Depth = %d\n", boundary, current_depth);

              // printf("Burst Memory sub_idx=%d\n", curr_subtree_idx);
              BURST_MEMORY_READ_VALUE:
              for(int i = 0; i < boundary; i++){
                nodes_value_buf[i] = nodes_value[subtree_node_value_list_offset + i];
              }
              BURST_MEMORY_READ_FEATURE:
              for(int i = 0; i < boundary; i++){
                nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + i];
              }
              SUBTREE_IDX_READ:
              for(int i = 0; i < boundary+1; i++){
                idx_to_subtree_buf[i] = idx_to_subtree[subtree_idx_to_subtree_offset + i];
              }

              //iterate over nodes in a subtree
              __attribute__((xcl_loop_tripcount(10, 20, 10)))
              SUBTREE_TRAVERSAL:
              for(int subtree_depth = 0; subtree_depth < current_depth; subtree_depth++){
                  __attribute__((xcl_loop_tripcount(375000, 375000, 375000)))
                  LEVEL_TRAVERSAL:
                  for(int tid = 0; tid < num_of_queries; tid++){
                  if(subtree_depth == 0 && curr_subtree_idx == 0){
                    query_data_buf[tid] = 0;
                    query_curr_subtree[tid] = 0;
                  }
                  // printf("Traverse Subnodes tid=%d, depth = %d\n", tid, subtree_depth);
                  unsigned query_entry = query_data_buf[tid];
                  unsigned row_offset = tid*num_of_features;
                  unsigned curr_node = query_entry >> 1;
                  
                  if(query_curr_subtree[tid] == curr_subtree_idx && !(query_entry & 0x1)){
                    unsigned short feature_buf = nodes_is_leaf_feature_id_buf[curr_node];
                    unsigned short feature_id = feature_buf >> 1;
                    float node_value    = nodes_value_buf[curr_node];
                    unsigned short is_tree_leaf    = feature_buf & 0x1;
                    if (is_tree_leaf==1){
                      // printf("query_results[%d] = %d, node_value = %f, subtree_idx = %d\n", tid, query_results[tid], node_value, query_curr_subtree[tid]);
                      query_results[tid] += node_value;
                      query_entry |= 0x00000001;
                    }
                    else{
                      bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                      //Mod
                      // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                      bool go_left = queries[row_offset + feature_id] <= node_value;
                      // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                      if (not_subtree_bottom){
                        // printf("Inside! curr_node = %d, boundary = %d\n", curr_node, subtree_leaf_idx_boundry);
                          // go to left child in subtree
                          if (go_left)
                              curr_node = curr_node*2 + 1;
                          // go to right child in subtree
                          else
                              curr_node = curr_node*2 + 2;
                          query_entry = curr_node << 1;
                      // if reach bottom of subtree, then we need to go to another subtree
                      } else{
                          unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                          unsigned subtree_offset;
                          // printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                          if (go_left)
                              subtree_offset = 2*leaf_idx;
                          else
                              subtree_offset = 2*leaf_idx+1;
                          query_curr_subtree[tid] = idx_to_subtree_buf[subtree_offset];
                          // printf("subtree_id = %d, leaf_bool = %d, tid = %d\n", query_curr_subtree[tid], query_data_buf[tid] & 0x1, tid);
                          query_entry = 0;
                          //stop the iterating of the current subtree, jump to the outer loop
                      }
                    }
                  }
                  query_data_buf[tid] = query_entry;
                }
// #endif
              //end subtree
              //if return from curr tree, skip all rest of subtrees, break from looping over subtrees
          }
        }
      }
    for(int tid = 0; tid < num_of_queries; tid++){
      results[tid] = query_results[tid] > num_of_trees/2;
    }
}
