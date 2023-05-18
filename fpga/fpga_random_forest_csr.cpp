// #ifdef FPGA_HIER
extern "C" {
  #ifndef __SYNTHESIS__
  #include <stdio.h>
  #endif
void csr_kernel(
  const unsigned num_of_trees           ,
  unsigned *   node_list_idx      ,
  unsigned *   edge_list_idx      ,
  unsigned *   node_is_leaf_idx   ,
  unsigned *   node_features_idx  ,
  unsigned *   node_values_idx    ,

  unsigned *   node_list_total    ,
  unsigned *   edge_list_total    ,
  unsigned *   node_is_leaf_total ,
  unsigned *   node_features_total,
  float    *   node_values_total  ,

  const unsigned start_tid        ,
  const unsigned num_of_queries         ,
  const unsigned num_of_features        ,
  float *queries                  ,
  unsigned *results                  
){
    for (int tid = start_tid; tid < num_of_queries; tid++){
      #pragma HLS LOOP_TRIPCOUNT min = 9000 max = 46875 avg = 25000
      //fetch a new query
      unsigned result_temp = 0;
      unsigned row_offset = tid*num_of_features;

         TREE_ITERATION:
         for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 100 avg = 50
            //go over trees

            unsigned node_list_offset = node_list_idx[tree_num];
            unsigned edge_list_offset = edge_list_idx[tree_num];
            unsigned node_is_leaf_offset = node_is_leaf_idx[tree_num];
            unsigned node_features_offset = node_features_idx[tree_num];
            unsigned node_values_offset = node_values_idx[tree_num];

            //start from node 0
            unsigned curr_node = 0;

            SUBTREE_ITERATION:
            //iterate over subtree
            while (true){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 35 avg = 15
              unsigned feature_id    = node_features_total[curr_node + node_features_offset]; 
              float node_value       = node_values_total[curr_node + node_values_offset];
              unsigned is_tree_leaf  = node_is_leaf_total[curr_node + node_is_leaf_offset];
              // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
              if (is_tree_leaf==1){
                result_temp += node_value;
                break;
              }
              // if node is not leaf, we need two comparisons to decide if we keep traverse 
              bool go_left = queries[row_offset + feature_id] <= node_value;
              if (go_left)
                  curr_node = edge_list_total[edge_list_offset + node_list_total[curr_node + node_list_offset]]; 
              // go to right child in subtree
              else
                  curr_node = edge_list_total[edge_list_offset + node_list_total[curr_node + node_list_offset]+1]; 
            }
          }
         unsigned threshold = num_of_trees/2;
         #ifndef __SYNTHESIS__
          // printf("tid = %d, result = %d\n", tid, result_temp);
         #endif
         results[tid] = result_temp > threshold;
        }
}
}