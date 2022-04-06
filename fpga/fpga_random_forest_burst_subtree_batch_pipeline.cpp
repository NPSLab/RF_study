// #ifdef FPGA_HIER
extern "C" {
  #ifndef __SYNTHESIS__
    #include <stdio.h>
    #endif
//#define MAX_SUBDEPTH 20
#define MAX_FEATURES 54
#define MAX_NODES ((1 << MAX_SUBDEPTH)-1)
#define MAX_BUFFER_SIZE 8192
const int c_size = MAX_NODES;
const int c_buf_size = MAX_BUFFER_SIZE;
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

  unsigned * query_curr_subtree             ,
  unsigned * query_data                     ,

  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  float    *  queries                     ,
  unsigned * results
){
  // U250 BRAM per SLR: 2.25 MB
  // U250 URAM per SLR: 11.52 MB
    #ifndef __SYNTHESIS__
    // printf("Kernel start!!!\n");
    #endif
  unsigned char results_buf[MAX_BUFFER_SIZE]; // 0.008 MB
  #pragma HLS bind_storage variable=results_buf type=RAM_2P impl=bram latency=1
  unsigned query_curr_subtree_buf[MAX_BUFFER_SIZE]; // 0.033 MB
  #pragma HLS bind_storage variable=query_curr_subtree_buf type=RAM_2P impl=bram latency=1
  unsigned query_data_buf[MAX_BUFFER_SIZE]; // 0.033 MB
  #pragma HLS bind_storage variable=query_data_buf type=RAM_2P impl=bram latency=1
  // #pragma HLS array_partition variable=query_data_buf type=cyclic factor=2
  float queries_buf[MAX_BUFFER_SIZE * MAX_FEATURES]; // 1.76MB
  #pragma HLS bind_storage variable=query_data_buf type=RAM_2P impl=bram latency=1
    TREE_ITERATION:
    for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
      #pragma HLS LOOP_TRIPCOUNT min = 100 max = 128 avg = 100

      RESET_BUFFERS:
      for(unsigned i = 0; i < num_of_queries; i++){
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 375000 avg = 275000
        query_data[i] = 0;
        query_curr_subtree[i] = 0;
      }

      //go over trees
      //tree_num=0;
      unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];

      //iterate over subtree
      #ifndef __SYNTHESIS__
      // printf("Tree Num=%d\n", tree_num);
      #endif
      SUBTREE_ITERATION:
      for(unsigned curr_subtree_idx = 0;
          curr_subtree_idx < prefix_sum_subtree_nums[tree_num+1] - tree_off_set;
          curr_subtree_idx++){
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 524288 avg = 10000
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
          unsigned short nodes_is_leaf_feature_id_buf[MAX_NODES]; // 2.097MB
          #pragma HLS bind_storage variable=nodes_is_leaf_feature_id_buf type=RAM_2P impl=uram latency=1
          float nodes_value_buf[MAX_NODES]; // 4.194MB
          #pragma HLS bind_storage variable=nodes_value_buf type=RAM_2P impl=uram latency=1
          unsigned idx_to_subtree_buf[MAX_NODES+1]; // 4.194MB
          #pragma HLS bind_storage variable=idx_to_subtree_buf type=RAM_2P impl=uram latency=1

          unsigned boundary = subtree_leaf_idx_boundry * 2 + 1;

          unsigned char current_depth = 1;
          unsigned boundary_temp = boundary;
          BOUNDARY_CALCULATION:
          for(current_depth; boundary_temp != 1; current_depth++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 20 avg = 10
            boundary_temp = boundary_temp >> 1;
          }

          #ifndef __SYNTHESIS__
          // printf("Subtree Index = %d, Boundary = %d, depth = %d\n", curr_subtree_idx, boundary, current_depth);
          #endif
          #ifndef __SYNTHESIS__
          // printf("Boundary = %d, Depth = %d\n", boundary, current_depth);
          #endif

          #ifndef __SYNTHESIS__
          // printf("Burst Memory sub_idx=%d\n", curr_subtree_idx);
          #endif
          BURST_MEMORY_READ_VALUE:
          for(int i = 0; i < boundary; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
            nodes_value_buf[i] = nodes_value[subtree_node_value_list_offset + i];
            #ifndef __SYNTHESIS__
            // if(i == 0) printf("nodes_value = %f", nodes_value[subtree_node_value_list_offset + i]);
            // if(i == 0) printf("nodes_value_buf = %f", nodes_value_buf[i]);
            #endif
          }
          BURST_MEMORY_READ_FEATURE:
          for(int i = 0; i < boundary; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
            nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[subtree_node_is_leaf_feature_id_list_offset + i];
          }
          SUBTREE_IDX_READ:
          for(int i = 0; i < boundary+1; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
            idx_to_subtree_buf[i] = idx_to_subtree[subtree_idx_to_subtree_offset + i];
          }

          //iterate over nodes in a subtree
          BATCH_TRAVERSAL:
          for(int tid = 0; tid < num_of_queries; tid += MAX_BUFFER_SIZE){
            #pragma HLS pipeline off
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 46 avg = 13
            unsigned short batch_size;
            // Calculate batch size
            if(tid + MAX_BUFFER_SIZE > num_of_queries){
              batch_size = num_of_queries - tid;
            }
            else{
              batch_size = MAX_BUFFER_SIZE;
            }
            // Load batch buffers
            BURST_RESULTS_READ:
            for(int i = 0; i < batch_size; i++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              results_buf[i] = results[tid + i];
            }
            BURST_CURR_SUBTREE_READ:
            for(int i = 0; i < batch_size; i++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              query_curr_subtree_buf[i] = query_curr_subtree[tid + i];
            }
            BURST_QUERY_DATA_READ:
            for(int i = 0; i < batch_size; i++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              query_data_buf[i] = query_data[tid + i];
            }
            BURST_QUERIES_WRITE:
            for(int i = 0; i < batch_size * num_of_features; i++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size * 54 avg = c_buf_size * 24
              queries_buf[i] = queries[tid*num_of_features + i];
            }

            // Traverse Subtree
            SUBTREE_TRAVERSAL:
            for(int subtree_depth = 0; subtree_depth < current_depth; subtree_depth++){
              #pragma HLS LOOP_TRIPCOUNT min = 10 max = 20 avg = 10
              #pragma HLS pipeline off
              LEVEL_TRAVERSAL:
              for(int bid = 0; bid < batch_size; bid++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              #pragma HLS pipeline II=1
			        // #pragma HLS unroll factor=2
              #pragma HLS DEPENDENCE variable=query_curr_subtree_buf inter RAW false
              #pragma HLS DEPENDENCE variable=query_data_buf inter RAW false
              #pragma HLS DEPENDENCE variable=results_buf inter RAW false
              #ifndef __SYNTHESIS__
              // if(tid + bid == 0)
              // printf("Traverse Subnodes tid=%d, depth = %d\n", tid + bid, subtree_depth);
              #endif
              unsigned query_entry;
              unsigned row_offset = bid*num_of_features;

              query_entry = query_data_buf[bid];

              unsigned curr_node = query_entry >> 1;

              if(query_curr_subtree_buf[bid] == curr_subtree_idx && !(query_entry & 0x1)){
                unsigned short feature_buf = nodes_is_leaf_feature_id_buf[curr_node];
                unsigned short feature_id = feature_buf >> 1;
                float node_value    = nodes_value_buf[curr_node];
                #ifndef __SYNTHESIS__
                // if(tid + bid == 0) printf("nodes_value_buf[%d] = %f", curr_node, nodes_value_buf[curr_node]);
                // if(tid + bid == 0) printf(" nodes_value_reg = %f", node_value);
                // if(tid + bid == 0) printf(" query_buf = %f, queries = %f\n", queries_buf[row_offset + feature_id], queries[(tid + bid) * num_of_features + feature_id]);
                #endif
                unsigned short is_tree_leaf    = feature_buf & 0x1;
                if (is_tree_leaf==1){
                  #ifndef __SYNTHESIS__
                  // if(tid + bid == 0) printf("query_results[%d] = %d, node_value = %f, subtree_idx = %d\n", bid, results_buf[bid], node_value, query_curr_subtree_buf[bid]\);
                  #endif
                  results_buf[bid] += nodes_value_buf[curr_node];
                  query_data_buf[bid] = query_entry | 0x00000001;
                }
                else{
                  bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                  //Mod
                  // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                  bool go_left = queries_buf[row_offset + feature_id] <= nodes_value_buf[curr_node];
                  // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                  if (not_subtree_bottom){
                    #ifndef __SYNTHESIS__
                    // if(tid + bid == 0) printf("Inside! curr_node = %d, query = %f, node_value = %f\n", curr_node, queries_buf[row_offset + feature_id], node_value);
                    #endif
                      // go to left child in subtree
                      if (go_left)
                          curr_node = curr_node*2 + 1;
                      // go to right child in subtree
                      else
                          curr_node = curr_node*2 + 2;
                      query_data_buf[bid] = curr_node << 1;
                  // if reach bottom of subtree, then we need to go to another subtree
                  } else{
                      unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                      unsigned subtree_offset;
                      #ifndef __SYNTHESIS__
                      // if(tid + bid == 0) printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                      #endif
                      if (go_left)
                          subtree_offset = 2*leaf_idx;
                      else
                          subtree_offset = 2*leaf_idx+1;
                      query_curr_subtree_buf[bid] = idx_to_subtree_buf[subtree_offset];
                      #ifndef __SYNTHESIS__
                      // if(tid + bid == 0) printf("subtree_id = %d, leaf_bool = %d, tid = %d\n", query_curr_subtree_buf[bid], query_data_buf[bid] & 0x1, tid + bid);
                      #endif
                      query_data_buf[bid] = 0;
                      //stop the iterating of the current subtree, jump to the outer loop
                  }
                }
              }
            }
          }
          // Write batch buffers
          BURST_RESULTS_WRITE:
          for(int i = 0; i < batch_size; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
            results[tid + i] = results_buf[i];
          }
          BURST_CURR_SUBTREE_WRITE:
          for(int i = 0; i < batch_size; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              query_curr_subtree[tid + i] = query_curr_subtree_buf[i];
          }
          BURST_QUERY_DATA_WRITE:
          for(int i = 0; i < batch_size; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              query_data[tid + i] = query_data_buf[i];
          }
          BURST_QUERIES_READ:
          for(int i = 0; i < batch_size; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_buf_size avg = c_buf_size
              queries[tid*num_of_features + i] = queries_buf[i];
          }
        }
      }
    }
    for(int tid = 0; tid < num_of_queries; tid++){
	    #pragma HLS LOOP_TRIPCOUNT min = 1 max = 350000 avg = 250000
      #pragma HLS DEPENDENCE variable=results inter RAW false
      #ifndef __SYNTHESIS__
        // if(tid == 0) printf("results[%d] = %d\n", tid, results[tid]);
      #endif
      results[tid] = results[tid] > num_of_trees/2;
    }
}
}
