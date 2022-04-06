// #ifdef FPGA_HIER
extern "C++" {
#ifndef __SYNTHESIS__
    // #include <stdio.h>
#endif
#include <hls_stream.h>
// #define MAX_SUBDEPTH 10
#define MAX_NODES ((1 << MAX_SUBDEPTH)-1)
#define MAX_FEATURES 54
const int c_size = MAX_NODES;

typedef struct {
   unsigned short query_results;
   unsigned int query_curr_subtree;
   unsigned int query_data_buf;
} query_t;

template<unsigned DEPTH>
struct rf_t {
  unsigned short nodes_is_leaf_feature_id_buf[(1 << (DEPTH + 1))-1]; // 2.097MB
  float nodes_value_buf[(1 << (DEPTH + 1))-1]; // 4.194MB
  unsigned idx_to_subtree_buf[(1 << (DEPTH + 1))]; // 4.194MB
};

void read_buffer(hls::stream<query_t>& read_stream, query_t * query_info,
                        unsigned num_of_features, hls::stream<float> * queries_stream,
                        float * queries, unsigned num_of_queries){
  for(int i = 0; i < num_of_queries; i++){
    unsigned row_offset = i*num_of_features;
    read_stream << query_info[i];
    for(int ii = 0; ii < num_of_features; ii++){
      if(ii < num_of_features){
        queries_stream[ii] << queries[row_offset + ii];
      }
    }
  }
}

void write_buffer(hls::stream<query_t>& write_stream, query_t * query_info, unsigned num_of_queries){
  for(int i = 0; i < num_of_queries; i++){
    write_stream >> query_info[i];
  }
}

template<unsigned DEPTH>
void process_level(hls::stream<query_t>& read_stream, hls::stream<query_t>& write_stream, rf_t<DEPTH> rf_buf,
                   hls::stream<float> * read_queries_stream,
				   hls::stream<float> * write_queries_stream,
                   unsigned char max_depth, unsigned subtree_leaf_idx_boundry,
                   unsigned num_of_queries, unsigned num_of_features){
  if(DEPTH < max_depth){
    for(int tid = 0; tid < num_of_queries; tid++){
      #ifndef __SYNTHESIS__
      //  if(tid == 0)
      //  printf("Traverse Subnodes tid=%d, depth = %d\n", tid, subtree_depth);
      #endif

      query_t query_stream_buf;
      float queries_buf[MAX_FEATURES];
      // #pragma HLS ARRAY_PARTITION variable=queries_buf type=complete

      // Load from FIFO buffer
      read_stream >> query_stream_buf;
      for(int i = 0; i < num_of_features; i++){
    	  read_queries_stream[i] >> queries_buf[i];
      }
      unsigned query_entry;

      query_entry = query_stream_buf.query_data_buf;

      unsigned curr_node = query_entry >> 1;

      unsigned depth_offset = curr_node - ((1 << DEPTH) - 1);

      if(query_stream_buf.query_curr_subtree == DEPTH && !(query_entry & 0x1)){
        unsigned short feature_buf = rf_buf.nodes_is_leaf_feature_id_buf[depth_offset];
        unsigned short feature_id = feature_buf >> 1;
        float node_value    = rf_buf.nodes_value_buf[depth_offset];
        unsigned short is_tree_leaf    = feature_buf & 0x1;
        if (is_tree_leaf==1){
          #ifndef __SYNTHESIS__
            //  if(tid == 0) printf("query_results[%d] = %d, node_value = %f, subtree_idx = %d\n", tid, query_results[tid], node_value, query_curr_subtree[tid]);
          #endif
          query_stream_buf.query_results += node_value;
          query_stream_buf.query_data_buf = query_entry | 0x00000001;
        }
        else{
          bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
          //Mod
          // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
          bool go_left = queries_buf[feature_id] <= node_value;
          // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
          if (not_subtree_bottom){
            #ifndef __SYNTHESIS__
              //  if(tid == 0) printf("Inside! curr_node = %d, boundary = %d, query = %f, node_value = %f\n", curr_node, subtree_leaf_idx_boundry, queries[row_offset + feature_id], node_value);
            #endif
              // go to left child in subtree
              if (go_left)
                  curr_node = curr_node*2 + 1;
              // go to right child in subtree
              else
                  curr_node = curr_node*2 + 2;
              query_stream_buf.query_data_buf = curr_node << 1;
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
              query_stream_buf.query_curr_subtree = rf_buf.idx_to_subtree_buf[subtree_offset];
              #ifndef __SYNTHESIS__
                //  if(tid == 0) printf("subtree_id = %d, leaf_bool = %d, tid = %d\n", query_curr_subtree[tid], query_data_buf[tid] & 0x1, tid);
              #endif
              query_stream_buf.query_data_buf = 0;
              //stop the iterating of the current subtree, jump to the outer loop
          }
        }
      }
      write_stream << query_stream_buf;
      for(int i = 0; i < num_of_features; i++){
	    write_queries_stream[i] << queries_buf[i];
	  }
    }
  }
}

template<unsigned DEPTH>
void process_level_last(hls::stream<query_t>& read_stream, hls::stream<query_t>& write_stream, rf_t<DEPTH> rf_buf,
                   hls::stream<float> * read_queries_stream,
                   unsigned char max_depth, unsigned subtree_leaf_idx_boundry,
                   unsigned num_of_queries, unsigned num_of_features){
  if(DEPTH < max_depth){
    for(int tid = 0; tid < num_of_queries; tid++){
      #ifndef __SYNTHESIS__
      //  if(tid == 0)
      //  printf("Traverse Subnodes tid=%d, depth = %d\n", tid, subtree_depth);
      #endif

      query_t query_stream_buf;
      float queries_buf[MAX_FEATURES];
      // #pragma HLS ARRAY_PARTITION variable=queries_buf type=complete

      // Load from FIFO buffer
      read_stream >> query_stream_buf;
      for(int i = 0; i < num_of_features; i++){
    	  read_queries_stream[i] >> queries_buf[i];
      }
      unsigned query_entry;

      query_entry = query_stream_buf.query_data_buf;

      unsigned curr_node = query_entry >> 1;

      unsigned depth_offset = curr_node - ((1 << DEPTH) - 1);

      if(query_stream_buf.query_curr_subtree == DEPTH && !(query_entry & 0x1)){
        unsigned short feature_buf = rf_buf.nodes_is_leaf_feature_id_buf[depth_offset];
        unsigned short feature_id = feature_buf >> 1;
        float node_value    = rf_buf.nodes_value_buf[depth_offset];
        unsigned short is_tree_leaf    = feature_buf & 0x1;
        if (is_tree_leaf==1){
          #ifndef __SYNTHESIS__
            //  if(tid == 0) printf("query_results[%d] = %d, node_value = %f, subtree_idx = %d\n", tid, query_results[tid], node_value, query_curr_subtree[tid]);
          #endif
          query_stream_buf.query_results += node_value;
          query_stream_buf.query_data_buf = query_entry | 0x00000001;
        }
        else{
          bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
          //Mod
          // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
          bool go_left = queries_buf[feature_id] <= node_value;
          // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
          if (not_subtree_bottom){
            #ifndef __SYNTHESIS__
              //  if(tid == 0) printf("Inside! curr_node = %d, boundary = %d, query = %f, node_value = %f\n", curr_node, subtree_leaf_idx_boundry, queries[row_offset + feature_id], node_value);
            #endif
              // go to left child in subtree
              if (go_left)
                  curr_node = curr_node*2 + 1;
              // go to right child in subtree
              else
                  curr_node = curr_node*2 + 2;
              query_stream_buf.query_data_buf = curr_node << 1;
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
              query_stream_buf.query_curr_subtree = rf_buf.idx_to_subtree_buf[subtree_offset];
              #ifndef __SYNTHESIS__
                //  if(tid == 0) printf("subtree_id = %d, leaf_bool = %d, tid = %d\n", query_curr_subtree[tid], query_data_buf[tid] & 0x1, tid);
              #endif
              query_stream_buf.query_data_buf = 0;
              //stop the iterating of the current subtree, jump to the outer loop
          }
        }
      }
      write_stream << query_stream_buf;
    }
  }
}

template <unsigned DEPTH>
void read_memory(rf_t<DEPTH> &rf_buf, unsigned int max_depth,
                 unsigned int subtree_node_is_leaf_feature_id_list_offset,
                 unsigned int subtree_node_value_list_offset,
                 unsigned int subtree_idx_to_subtree_offset,
                 unsigned short * nodes_is_leaf_feature_id,
                 float * nodes_value,
                 unsigned int * idx_to_subtree){
  // #pragma HLS INLINE
  if(DEPTH < max_depth){
    unsigned i_offset = (1 << DEPTH) - 1;
    for(int i = 0; i < (1 << DEPTH) - 1; i++){
      rf_buf.nodes_is_leaf_feature_id_buf[i] = nodes_is_leaf_feature_id[i_offset + subtree_node_is_leaf_feature_id_list_offset + i];
      rf_buf.nodes_value_buf[i] = nodes_value[i_offset + subtree_node_value_list_offset + i];
      rf_buf.idx_to_subtree_buf[i] = idx_to_subtree[i_offset + subtree_idx_to_subtree_offset + i];
    }
    rf_buf.idx_to_subtree_buf[(1 << DEPTH) - 1] = idx_to_subtree[i_offset + subtree_idx_to_subtree_offset + (1 << DEPTH) - 1];
  }
}

void read_process_write_loop(query_t * query_info, unsigned num_of_features,
		float * queries, unsigned num_of_queries, rf_t<0> rf_buffer_0, rf_t<1> rf_buffer_1,
		rf_t<2> rf_buffer_2, rf_t<3> rf_buffer_3, rf_t<4> rf_buffer_4, rf_t<5> rf_buffer_5,
		rf_t<6> rf_buffer_6, rf_t<7> rf_buffer_7, rf_t<8> rf_buffer_8, rf_t<9> rf_buffer_9,
		// rf_t<10> rf_buffer_10,
		// rf_t<11> rf_buffer_11, rf_t<12> rf_buffer_12, rf_t<13> rf_buffer_13, rf_t<14> rf_buffer_14,
		// rf_t<15> rf_buffer_15, rf_t<16> rf_buffer_16, rf_t<17> rf_buffer_17, rf_t<18> rf_buffer_18,
		// rf_t<19> rf_buffer_19,
		unsigned current_depth, unsigned subtree_leaf_idx_boundry){
	hls::stream<query_t> query_stream[MAX_SUBDEPTH+1];
	hls::stream<float> queries_stream[MAX_SUBDEPTH][MAX_FEATURES];
	#pragma HLS ARRAY_PARTITION variable=queries_stream type=complete dim=2

	#pragma HLS dataflow

	read_buffer(query_stream[0], query_info,
				num_of_features, queries_stream[0],
				queries, num_of_queries);

	process_level<0>(query_stream[0], query_stream[1], rf_buffer_0, queries_stream[0], queries_stream[1], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
  process_level<1>(query_stream[1], query_stream[2], rf_buffer_1, queries_stream[1], queries_stream[2], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<2>(query_stream[2], query_stream[3], rf_buffer_2, queries_stream[2], queries_stream[3], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<3>(query_stream[3], query_stream[4], rf_buffer_3, queries_stream[3], queries_stream[4], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<4>(query_stream[4], query_stream[5], rf_buffer_4, queries_stream[4], queries_stream[5], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<5>(query_stream[5], query_stream[6], rf_buffer_5, queries_stream[5], queries_stream[6], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<6>(query_stream[6], query_stream[7], rf_buffer_6, queries_stream[6], queries_stream[7], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<7>(query_stream[7], query_stream[8], rf_buffer_7, queries_stream[7], queries_stream[8], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level<8>(query_stream[8], query_stream[9], rf_buffer_8, queries_stream[8], queries_stream[9], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	process_level_last<9>(query_stream[9], query_stream[10], rf_buffer_9, queries_stream[9], /*queries_stream[10],*/ current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<10>(query_stream[10], query_stream[11], rf_buffer_10, queries_stream[10], queries_stream[11], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<11>(query_stream[11], query_stream[12], rf_buffer_11, queries_stream[11], queries_stream[12], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<12>(query_stream[12], query_stream[13], rf_buffer_12, queries_stream[12], queries_stream[13], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<13>(query_stream[13], query_stream[14], rf_buffer_13, queries_stream[13], queries_stream[14], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<14>(query_stream[14], query_stream[15], rf_buffer_14, queries_stream[14], queries_stream[15], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<15>(query_stream[15], query_stream[16], rf_buffer_15, queries_stream[15], queries_stream[16], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<16>(query_stream[16], query_stream[17], rf_buffer_16, queries_stream[16], queries_stream[17], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<17>(query_stream[17], query_stream[18], rf_buffer_17, queries_stream[17], queries_stream[18], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level<18>(query_stream[18], query_stream[19], rf_buffer_18, queries_stream[18], queries_stream[19], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);
	// process_level_last<19>(query_stream[19], query_stream[20], rf_buffer_19, queries_stream[19], current_depth, subtree_leaf_idx_boundry, num_of_queries, num_of_features);

	write_buffer(query_stream[10], query_info, num_of_queries);
}

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

  const unsigned num_of_queries                  ,
  const unsigned num_of_features                 ,
  float    *  queries                     ,
  unsigned * results
){
  #pragma HLS INTERFACE m_axi port=prefix_sum_subtree_nums offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=nodes_value offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=nodes_is_leaf_feature_id offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=idx_to_subtree offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=g_subtree_nodes_offset offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=g_subtree_idx_to_subtree_offset offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=queries offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=results offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=query_info offset=slave bundle=gmem1
  // U250 BRAM per SLR: 2.25 MB
  // U250 URAM per SLR: 11.52 MB
    #ifndef __SYNTHESIS__
      //  printf("Kernel start!!!\n");
    #endif
	// #pragma HLS DATAFLOW
        TREE_ITERATION:
        for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){
          #pragma HLS LOOP_TRIPCOUNT min = 100 max = 128 avg = 100

          RESET_BUFFERS:
          for(unsigned i = 0; i < num_of_queries; i++){
            #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1000000 avg = 375000
            query_info[i].query_data_buf = 0;
            query_info[i].query_curr_subtree = 0;
          }

          //go over trees
          //tree_num=0;
          unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];

          //iterate over subtree
          #ifndef __SYNTHESIS__
            //  printf("Tree Num=%d\n", tree_num);
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

            unsigned boundary = subtree_leaf_idx_boundry * 2 + 1;

            unsigned char current_depth = 1;
            unsigned boundary_temp = boundary;
            BOUNDARY_CALCULATION:
            for(current_depth; boundary_temp != 1; current_depth++){
              #pragma HLS LOOP_TRIPCOUNT min = 1 max = 20 avg = 10
              boundary_temp = boundary_temp >> 1;
            }

            rf_t<0> rf_buffer_0;
            rf_t<1> rf_buffer_1;
            rf_t<2> rf_buffer_2;
            rf_t<3> rf_buffer_3;
            rf_t<4> rf_buffer_4;
            rf_t<5> rf_buffer_5;
            rf_t<6> rf_buffer_6;
            rf_t<7> rf_buffer_7;
            rf_t<8> rf_buffer_8;
            rf_t<9> rf_buffer_9;
            // rf_t<10> rf_buffer_10;
            // rf_t<11> rf_buffer_11;
            // rf_t<12> rf_buffer_12;
            // rf_t<13> rf_buffer_13;
            // rf_t<14> rf_buffer_14;
            // rf_t<15> rf_buffer_15;
            // rf_t<16> rf_buffer_16;
            // rf_t<17> rf_buffer_17;
            // rf_t<18> rf_buffer_18;
            // rf_t<19> rf_buffer_19;

            read_memory<0>(rf_buffer_0, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<1>(rf_buffer_1, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<2>(rf_buffer_2, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<3>(rf_buffer_3, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<4>(rf_buffer_4, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<5>(rf_buffer_5, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<6>(rf_buffer_6, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<7>(rf_buffer_7, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<8>(rf_buffer_8, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            read_memory<9>(rf_buffer_9, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<10>(rf_buffer_10, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<11>(rf_buffer_11, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<12>(rf_buffer_12, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<13>(rf_buffer_13, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<14>(rf_buffer_14, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<15>(rf_buffer_15, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<16>(rf_buffer_16, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<17>(rf_buffer_17, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<18>(rf_buffer_18, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);
            // read_memory<19>(rf_buffer_19, current_depth, subtree_node_is_leaf_feature_id_list_offset, subtree_node_value_list_offset, subtree_idx_to_subtree_offset, nodes_is_leaf_feature_id, nodes_value, idx_to_subtree);

            read_process_write_loop(query_info, num_of_features,
            		queries, num_of_queries, rf_buffer_0, rf_buffer_1,
					rf_buffer_2, rf_buffer_3, rf_buffer_4, rf_buffer_5,
					rf_buffer_6, rf_buffer_7, rf_buffer_8, rf_buffer_9,
					// rf_buffer_10, rf_buffer_11, rf_buffer_12, rf_buffer_13,
					// rf_buffer_14, rf_buffer_15, rf_buffer_16, rf_buffer_17,
					// rf_buffer_18, rf_buffer_19,
            		current_depth, subtree_leaf_idx_boundry);


        }
      }
    for(int tid = 0; tid < num_of_queries; tid++){
      #pragma HLS LOOP_TRIPCOUNT min = 1 max = 350000 avg = 250000
      #ifndef __SYNTHESIS__
        // if(tid == 0) printf("results[%d] = %d\n", tid, query_results[tid]);
      #endif
      results[tid] = query_info[tid].query_results > num_of_trees/2;
    }
}
}
