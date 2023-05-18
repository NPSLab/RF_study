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

    for (int tid = 0; tid < num_of_queries; tid++){
      // printf("New Query tid=%d\n", tid);
      //fetch a new query
      unsigned result_temp = 0;
      unsigned row_offset = tid*num_of_features;
         __attribute__((opencl_unroll_hint(16))) TREE_ITERATION:
         for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
            //go over trees
            //tree_num=0;
            unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
            //unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;
        
            unsigned curr_subtree_idx = 0 ; 

            bool return_from_tree = false;
            //iterate over subtree 
            // printf("Subtree Iteration tid=%d\n", tid);
            // printf("Tree Num=%d\n", tree_num);
            SUBTREE_ITERATION:
            while (!return_from_tree){
              // printf("Subtree Index = %d\n", curr_subtree_idx);
                //fetch the subtree nodes
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
                bool return_from_curr_tree[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                unsigned short nodes_is_leaf_feature_id_buf[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                float nodes_value_buf[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                float queries_buf[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                // bool not_subtree_bottom[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                unsigned short go_left[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));
                float results_buf[MAX_SUBDEPTH] __attribute__((xcl_array_partition(complete, 1)));

                // printf("Burst Memory tid=%d\n", tid);
                BURST_MEMORY:
                for(int i = 0; i < MAX_SUBDEPTH; i++){
                  nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + i];
                  nodes_value_buf[i] = nodes_value[subtree_node_value_list_offset + i];
                }

                //iterate over nodes in a subtree
                // printf("Traverse Subnodes tid=%d\n", tid);
                SUBTREE_TRAVERSAL:
                for (unsigned subnode_id = 0; subnode_id < subtree_batch_boundary; subnode_id++){
                  //Mod
                    unsigned short feature_id = nodes_is_leaf_feature_id_buf[subnode_id] >> 1;                    
                    float node_value    = nodes_value_buf[subnode_id];
                    go_left[subnode_id] = queries[row_offset + feature_id] <= node_value;
                    bool is_tree_leaf    = nodes_is_leaf_feature_id_buf[subnode_id] & 0x1;
                    // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                    if (is_tree_leaf==1){
                      // printf("Found Leaf!!! tid=%d, subnode_id = %d\n", tid, subnode_id);
                      results_buf[subnode_id] = node_value;
                      return_from_curr_tree[subnode_id] = true;
                    } //goto SUBTREE_END;
                    else{
                      results_buf[subnode_id] = 0;
                      return_from_curr_tree[subnode_id] = false;
                    }
                    // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                    // not_subtree_bottom[subnode_id] = subnode_id < subtree_leaf_idx_boundry;
                }
                // Compute subnode path

                bool subtree_bottom = false;
                //start from node 0
                unsigned curr_node = 0;
                // printf("Path Subtree tid=%d\n", tid);
                SUBTREE_PATH:
                while(!subtree_bottom && !return_from_curr_tree[curr_node]){
                  // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                  if (curr_node < subtree_leaf_idx_boundry){
                      // printf("Inside! curr_node = %d, boundary = %d\n", curr_node, subtree_leaf_idx_boundry);
                      // go to left child in subtree
                      if (go_left[curr_node])
                          curr_node = curr_node*2 + 1;
                      // go to right child in subtree
                      else
                          curr_node = curr_node*2 + 2;
                  // if reach bottom of subtree, then we need to go to another subtree
                  } else{
                      unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                      // printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                      if (go_left[curr_node])
                          curr_subtree_idx = idx_to_subtree[subtree_idx_to_subtree_offset + 2*leaf_idx];
                      else
                          curr_subtree_idx = idx_to_subtree[subtree_idx_to_subtree_offset + 2*leaf_idx+1];
                      //stop the iterating of the current subtree, leave while statement
                      subtree_bottom = true;
                  }
                }
                //end subtree
                //if return from curr tree, skip all rest of subtrees, stop looping over subtrees
                return_from_tree = return_from_curr_tree[curr_node];
                result_temp += results_buf[curr_node];
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
