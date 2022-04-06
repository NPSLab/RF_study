// #ifdef FPGA_HIER
extern "C" {
#ifndef __SYNTHESIS__
    #include <stdio.h>
#endif
#include "assert.h"
// #define MAX_SUBDEPTH 18
#define MAX_FEATURES 1024
#define MAX_NODES ((1 << MAX_SUBDEPTH)-1)
const int c_size = MAX_NODES;

typedef struct {
   unsigned short query_results;
   unsigned int query_curr_subtree;
   unsigned int query_data_buf;
} query_t;

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

  query_t * query_info                      ,

  const unsigned start_tid                  ,
  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  float    *  queries                     ,
  unsigned * results
){
  // U250 BRAM per SLR: 2.25 MB
  // U250 URAM per SLR: 11.52 MB
    #ifndef __SYNTHESIS__
      //  printf("Kernel start!!!\n");
    #endif
	// #pragma HLS DATAFLOW
        TREE_ITERATION:
        for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
          #pragma HLS LOOP_TRIPCOUNT min = 100 max = 128 avg = 100

          //go over trees
          //tree_num=0;
          unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];

          //iterate over subtree
          #ifndef __SYNTHESIS__
            //  printf("Tree Num=%d\n", tree_num);
          #endif
          SUBTREE_ITERATION:
          unsigned curr_subtree_idx = 0;
            //fetch the subtree nodes
          //Mod

            unsigned subtree_node_value_list_offset
                = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

            unsigned subtree_node_is_leaf_feature_id_list_offset
                = g_subtree_nodes_offset[tree_off_set+curr_subtree_idx];

            //fetch subtree_leaf_idx_boundry
            unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+curr_subtree_idx];

            //fetch subtree_idx_to_other_subtree
            unsigned subtree_idx_to_subtree_offset = g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

            // Burst access data structures
            unsigned short nodes_is_leaf_feature_id_buf[MAX_NODES]; // 2.097MB
            // #pragma HLS bind_storage variable=nodes_is_leaf_feature_id_buf type=RAM_1P impl=bram
            float nodes_value_buf[MAX_NODES]; // 4.194MB
            #pragma HLS bind_storage variable=nodes_value_buf type=RAM_1P impl=uram
            unsigned idx_to_subtree_buf[MAX_NODES+1]; // 4.194MB
            #pragma HLS bind_storage variable=idx_to_subtree_buf type=RAM_1P impl=uram

            unsigned boundary = subtree_leaf_idx_boundry * 2 + 1;

            unsigned char current_depth = 1;
            unsigned boundary_temp = boundary;
            BOUNDARY_CALCULATION:
            for(current_depth; boundary_temp != 1; current_depth++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 20 avg = 10
              boundary_temp = boundary_temp >> 1;
            }

            #ifndef __SYNTHESIS__
              //  printf("Subtree Index = %d, Boundary = %d, depth = %d\n", curr_subtree_idx, boundary, current_depth);
            #endif
            #ifndef __SYNTHESIS__
              //  printf("Boundary = %d, Depth = %d\n", boundary, current_depth);
            #endif

            #ifndef __SYNTHESIS__
              //  printf("Burst Memory sub_idx=%d\n", curr_subtree_idx);
            #endif
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
            SUBTREE_IDX_READ:
            for(int i = 0; i < boundary+1; i++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = c_size avg = 1024
              idx_to_subtree_buf[i] = idx_to_subtree[subtree_idx_to_subtree_offset + i];
            }

            //iterate over nodes in a subtree
            BATCH_SUBTREE_TRAVERSAL:
            for(int subtree_depth = 0; subtree_depth < current_depth; subtree_depth++){
              #pragma HLS LOOP_TRIPCOUNT min = 10 max = 20 avg = 10
              #pragma HLS pipeline off
              LEVEL_TRAVERSAL:
              for(int tid = start_tid; tid < num_of_queries; tid++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 350000 avg = 250000
              #pragma HLS pipeline II=2
              #pragma HLS DEPENDENCE variable=query_info inter RAW false
              #ifndef __SYNTHESIS__
                //  if(tid == 0)
                //  printf("Traverse Subnodes tid=%d, depth = %d\n", tid, subtree_depth);
              #endif
              unsigned query_entry;
              unsigned row_offset = tid*num_of_features;
              unsigned query_offset = tid*num_of_trees + tree_num;

              query_t query_buf;

              if(subtree_depth == 0){
                query_buf.query_results = 0;
                query_buf.query_curr_subtree = 0;
                query_buf.query_data_buf = 0;
              }
              else{
                query_buf = query_info[query_offset];
                #ifndef __SYNTHESIS__
                  // if(tid == 8) printf("else curr_node = %d, %d\n", query_buf.query_data_buf >> 1, query_info[query_offset].query_data_buf >> 1);
                #endif
              }

              query_entry = query_buf.query_data_buf;

              #ifndef __SYNTHESIS__
                  // if(tid == 8) printf("super before curr_node = %d, %d\n", query_entry >> 1, query_info[query_offset].query_data_buf >> 1);
              #endif

              unsigned curr_node = query_entry >> 1;
              #ifndef __SYNTHESIS__
                  // if(tid == 8) printf("before curr_node = %d, %d\n", curr_node, query_info[query_offset].query_data_buf >> 1);
              #endif
              if(query_buf.query_curr_subtree == curr_subtree_idx && !(query_entry & 0x1)){
                unsigned short feature_buf = nodes_is_leaf_feature_id_buf[curr_node];
                unsigned short feature_id = feature_buf >> 1;
                float node_value    = nodes_value_buf[curr_node];
                unsigned short is_tree_leaf    = feature_buf & 0x1;
                if (is_tree_leaf==1){
                  #ifndef __SYNTHESIS__
                    //  if(tid == 0) printf("query_results[%d] = %d, node_value = %f, subtree_idx = %d\n", tid, query_results[tid], node_value, query_curr_subtree[tid]);
                    // printf("Leaf found! [%d] = %d\n", tid, node_value);
                  #endif
                  query_buf.query_results = node_value;
                  query_buf.query_data_buf = query_entry | 0x00000001;
                }
                else{
                  bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                  //Mod
                  // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                  bool go_left = queries[row_offset + feature_id] <= node_value;
                  // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                  if (not_subtree_bottom){
                      // go to left child in subtree
                      if (go_left)
                          curr_node = curr_node*2 + 1;
                      // go to right child in subtree
                      else
                          curr_node = curr_node*2 + 2;
                      query_buf.query_data_buf = curr_node << 1;
                      #ifndef __SYNTHESIS__
                        // if(tid == 8) printf("Inside! curr_node = %d, boundary = %d, query = %f, node_value = %f\n", curr_node, subtree_leaf_idx_boundry, queries[row_offset + feature_id], node_value);
                      #endif
                  // if reach bottom of subtree, then we need to go to another subtree
                  } else{
                      unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                      unsigned subtree_offset;
                      #ifndef __SYNTHESIS__
                        //  if(tid == 0) printf("Outside! curr_node = %d, boundary = %d, leaf_idx = %d\n", curr_node, subtree_leaf_idx_boundry, leaf_idx);
                      #endif
                      if (go_left)
                          subtree_offset = 2*leaf_idx;
                      else
                          subtree_offset = 2*leaf_idx+1;
                      query_buf.query_curr_subtree = idx_to_subtree_buf[subtree_offset];
                      #ifndef __SYNTHESIS__
                        // printf("subtree_id = %d, curr_node = %d, check = %d, tid = %d, subdepth = %d, max_depth = %d\n", query_buf.query_curr_subtree, curr_node, query_info[query_offset].query_data_buf >> 1, tid, subtree_depth, current_depth);
                      #endif
                      query_buf.query_data_buf = 0;
                      //stop the iterating of the current subtree, jump to the outer loop
                  }
                }
              }
              query_info[query_offset] = query_buf;
              #ifndef __SYNTHESIS__
                // if(tid == 8) printf("curr_node = %d, %d\n", query_buf.query_data_buf >> 1, query_info[query_offset].query_data_buf >> 1);
              #endif
            }
// #endif
            //end subtree
            //if return from curr tree, skip all rest of subtrees, break from looping over subtrees
        }
      }
      ITER_QUERY:
      for(int tid = start_tid; tid < num_of_queries; tid++){
      #pragma HLS LOOP_TRIPCOUNT min = 9000 max = 46875 avg = 25000
        unsigned result_temp = 0;
        unsigned row_offset = tid*num_of_features;
        float queries_buf[MAX_FEATURES];
        for(int i = 0; i < num_of_features; i++){
          #pragma HLS LOOP_TRIPCOUNT min = 1 max = 54 avg = 25
          queries_buf[i] = queries[row_offset + i];
        }
        ITER_TREE_ITERATION:
        for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
          unsigned query_offset = tid*num_of_trees + tree_num;
          query_t query_buf = query_info[query_offset];
          unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
          unsigned curr_subtree_idx = query_buf.query_curr_subtree;
          bool return_from_subtree = false;
          unsigned subtree_offset = 0;

          #ifndef __SYNTHESIS__
              // printf("Query Info! [%d] = %d, output = %d\n", tid, query_buf.query_data_buf, query_buf.query_data_buf & 0x1);
              // printf("Curr Subtree Idx = %d\n", curr_subtree_idx);
          #endif
          
          if(!(query_buf.query_data_buf & 0x1)){
            ITER_SUBTREE_ITER:
            while (true){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 2
              // if(tid == 0 && tree_num == 0) printf("Subtree Index = %d\n", curr_subtree_idx);
                //fetch the subtree nodes
              unsigned curr_node;
              if(return_from_subtree){
                curr_subtree_idx = idx_to_subtree[subtree_offset];
                return_from_subtree = false;
                //start from node 0
                curr_node = 0;
              }
              else{
                //start from node 0
                curr_node = query_buf.query_data_buf >> 1;
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

                ITER_SUBTREE_TRAVERSAL:
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
          else{
            #ifndef __SYNTHESIS__
            if(tree_num == 0)
              // printf("Result found! [%d] = %d\n", tid, query_buf.query_results);
            #endif
            result_temp += query_buf.query_results;
          }
        }
        #ifndef __SYNTHESIS__
          // printf("result_temp[%d] = %d\n", tid, result_temp);
        #endif
        results[tid] = result_temp > num_of_trees/2;
      }
  }
}
