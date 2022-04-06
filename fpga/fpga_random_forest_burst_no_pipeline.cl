// #ifdef FPGA_HIER
// #define MAX_SUBDEPTH 15
__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_zero_global_work_offset))
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
	__attribute__((xcl_loop_tripcount(100000, 100000, 100000)))
    for (int tid = 0; tid < num_of_queries; tid++){
      // printf("New Query tid=%d\n", tid);
      //fetch a new query
      unsigned result_temp = 0;
      unsigned row_offset = tid*num_of_features;
      float queries_buf[1024];
      __attribute__((xcl_loop_tripcount(10, 100, 50)))
      for(int i = 0; i < num_of_features; i++){
        queries_buf[i] = queries[row_offset + i];
      }
      __attribute__((xcl_loop_tripcount(100, 100, 100)))
         TREE_ITERATION:
         for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
            //go over trees
            //tree_num=0;
            unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
            //unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;

            unsigned curr_subtree_idx = 0 ;
            unsigned subtree_offset = 0;
            bool update_subtree = false;

            //iterate over subtree
            // printf("Subtree Iteration tid=%d\n", tid);
            // printf("Tree Num=%d\n", tree_num);
            __attribute__((xcl_loop_tripcount(1, 3, 3)))
            SUBTREE_ITERATION:
            while (true){
                //fetch the subtree nodes
              //Mod

              if(update_subtree){
                curr_subtree_idx = idx_to_subtree[subtree_offset];
                update_subtree = false;
              }

              // printf("Subtree Index = %d, Offset = %d\n", curr_subtree_idx, subtree_offset);

                const unsigned subtree_node_value_list_offset
                    = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

                const unsigned subtree_node_is_leaf_feature_id_list_offset
                    = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

                //fetch subtree_leaf_idx_boundry
                const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+curr_subtree_idx];

                //fetch subtree_idx_to_other_subtree
                const unsigned subtree_idx_to_subtree_offset = g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

                // Burst access data structures
                unsigned short nodes_is_leaf_feature_id_buf[MAX_SUBDEPTH];
                float nodes_value_buf[MAX_SUBDEPTH];

                bool return_from_curr_tree = false;

                //start from node 0
                unsigned curr_node = 0;

                // printf("Burst Memory tid=%d\n", tid);
                BURST_MEMORY_READ:
                for(int i = 0; i < MAX_SUBDEPTH; i++){
                  nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + i];
                  nodes_value_buf[i] = nodes_value[subtree_node_value_list_offset + i];
                }

                //iterate over nodes in a subtree
                // printf("Traverse Subnodes tid=%d\n", tid);
                __attribute__((xcl_loop_tripcount(1, 3, 3)))
                SUBTREE_TRAVERSAL:
                while (true){
                  //Mod
                    unsigned short feature_id = nodes_is_leaf_feature_id_buf[curr_node] >> 1;
                    float node_value    = nodes_value_buf[curr_node];
                    unsigned short is_tree_leaf    = nodes_is_leaf_feature_id_buf[curr_node] & 0x1;
                    // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                    //if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); return_from_curr_tree = true; break; }
                    // if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); return_from_curr_tree = true; goto SUBTREE_END; }
                    if (is_tree_leaf==1){ result_temp += node_value; return_from_curr_tree = true; goto SUBTREE_END; }
                    // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                    bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                    bool go_left = queries_buf[feature_id] <= node_value;
                    // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                    if (not_subtree_bottom){
                      // printf("Inside! curr_node = %d, boundary = %d\n", curr_node, subtree_leaf_idx_boundry);
                        // go to left child in subtree
                        if (go_left)
                            curr_node = curr_node*2 + 1;
                        // go to right child in subtree
                        else
                            curr_node = curr_node*2 + 2;
                    // if reach bottom of subtree, then we need to go to another subtree
                    } else{
                        unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                        // printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                        update_subtree = true;
                        if (go_left)
                            subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx;
                        else
                            subtree_offset = subtree_idx_to_subtree_offset + 2*leaf_idx+1;
                        //stop the iterating of the current subtree, jump to the outer loop
                        //break;
                        goto SUBTREE_END;
                    }
                }
// #endif
                //end subtree
                //if return from curr tree, skip all rest of subtrees, break from looping over subtrees
SUBTREE_END:
                if (return_from_curr_tree) break;
            }
          }
      results[tid] = result_temp;
    }
}
// #endif

// #ifdef FPGA_CSR
// __global__ void
// csr_kernel(
//   unsigned num_of_trees           ,
//   unsigned *   node_list_idx      ,
//   unsigned *   edge_list_idx      ,
//   unsigned *   node_is_leaf_idx   ,
//   unsigned *   node_features_idx  ,
//   unsigned *   node_values_idx    ,

//   unsigned *   node_list_total    ,
//   unsigned *   edge_list_total    ,
//   unsigned *   node_is_leaf_total ,
//   unsigned *   node_features_total,
//   float    *   node_values_total  ,

//   unsigned num_of_queries         ,
//   unsigned num_of_features        ,
//   float *queries                  ,
//   unsigned *results
// ){
//     for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < num_of_queries; tid += blockDim.x*gridDim.x){
//             //fetch a new query
//             float * row = queries + tid*num_of_features;
//             //go over trees
//             for(int i=0; i< num_of_trees; ++i){
//                 //csr layout
//                 //unsigned num_of_nodes = node_list_idx[i+1]-node_list_idx[i]-1;
//                 unsigned * node_list = node_list_total + node_list_idx[i];
//                 unsigned * edge_list = edge_list_total + edge_list_idx[i];
//                 unsigned * node_is_leaf = node_is_leaf_total + node_is_leaf_idx[i];
//                 unsigned * node_features = node_features_total + node_features_idx[i];
//                 float * node_values = node_values_total + node_values_idx[i];

//                 //start from node 0
//                 unsigned curr_node = 0;
//                 //iterate over nodes in a subtree
//                 while (true){
//                     unsigned feature_id    = node_features[curr_node];
//                     float node_value       = node_values[curr_node];
//                     unsigned is_tree_leaf  = node_is_leaf[curr_node];
//                     // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
//                     if (is_tree_leaf==1){
//                       //results[tid] = node_value;
// //                        if (node_value == 1.0f){
//                           atomicAdd(results+tid,(unsigned)node_value);
// //                        }
//                       break;
//                     }
//                     // if node is not leaf, we need two comparisons to decide if we keep traverse
//                     bool go_left = row[feature_id] <= node_value;
//                     if (go_left)
//                         curr_node = edge_list[node_list[curr_node]];
//                     // go to right child in subtree
//                     else
//                         curr_node = edge_list[node_list[curr_node]+1];
//                 }
//             }
//     }
// }
// #endif

__kernel void
generate_results(const unsigned num_of_queries, const unsigned num_of_trees, global unsigned * results){
    unsigned threshold = num_of_trees/2;
    for (int tid = get_global_id(0); tid < num_of_queries; tid += get_local_size(0) * get_num_groups(0)){
      if (results[tid] > threshold)
        results[tid] = 1;
      else
        results[tid] = 0;
    }
}
