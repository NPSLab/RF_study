#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define GPU_HIER
#define ITER
// #define ET5

/* different GPU kernel versions 

#define GPU_CSR
#define GPU_HIER
//if GPU_HIER is defined, then at least one of the following needs to be defined
#define ITER
#define ET2
#define ET3
#define ET4
#define ET5

*/

#define QUERY_LOAD_1
#define QUERY_LOAD_2

#define RATIO_QUERIES 0.5
#define SHARED_MEM_USAGE 1.0

using namespace std;

#define TIMING

#ifdef TIMING
  #define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
  #define START_TIMER  start = std::chrono::high_resolution_clock::now();
  #define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
        std::chrono::duration_cast<std::chrono::milliseconds>( \
                     std::chrono::high_resolution_clock::now()-start \
              ).count() << " ms " << std::endl; 
#else
  #define INIT_TIMER
  #define START_TIMER
  #define STOP_TIMER(name)
#endif

#ifdef GPU_HIER
__global__ void
hier_kernel(
  unsigned num_of_trees           ,
  unsigned *prefix_sum_subtree_nums        ,
  float    *nodes                          ,
  unsigned *idx_to_subtree                 ,  
  unsigned *leaf_idx_boundry               ,
  unsigned *g_subtree_nodes_offset         ,  
  unsigned *g_subtree_idx_to_subtree_offset,  

  unsigned num_of_queries         ,
  unsigned num_of_features        ,
  float *queries                  ,
  unsigned *results               ,
  unsigned long sm_bytes_available,
  float ratio_queries             ,
  int max_st_depth                
){
    unsigned long queries_allocable = floor((sm_bytes_available*ratio_queries)/(sizeof(float)*num_of_features));
    unsigned long subtree_bytes_allocable = floor(sm_bytes_available*(1.0-ratio_queries));
    int nodes_per_subtree = pow(2.0,max_st_depth)-1;
    int max_subtrees_loadable = floor(subtree_bytes_allocable/(3*sizeof(float)*nodes_per_subtree));

    extern __shared__ float shared_mem_space[];
    float* query_space = shared_mem_space;
    float* subtree_space = shared_mem_space+(queries_allocable*num_of_features);

    // __shared__ float query_space[queries_allocable*num_of_features];
    // queries + tid*num_of_features
    int queries_per_block = ceil((double)num_of_queries/(double)gridDim.x);
    int batch_start = queries_per_block*blockIdx.x;
    int batch_end;
    if(blockIdx.x == gridDim.x-1){
      batch_end = num_of_queries; 
    }else{
      batch_end = batch_start + queries_per_block;
    }

    int current_query = batch_start;
    while(current_query < batch_end){
      //fetch a new query
      unsigned int queries_used = 0;

      // LOAD QUERIES INTO SHARED MEM
      #ifdef QUERY_LOAD_2
      while(queries_used < queries_allocable){
        for (int fn = threadIdx.x; fn < num_of_features; fn+=blockDim.x)
        {
          query_space[queries_used*num_of_features+fn] = queries[(queries_used+current_query)*num_of_features+fn];
        }
        queries_used++;
      }
      #endif
      __syncthreads();

      for (int tid = current_query+threadIdx.x; tid < current_query+queries_used; tid+=blockDim.x)            // iterate over all queries in shared mem cache
      {
        float * row = query_space + (tid-current_query)*num_of_features;                                      // assign a query to a thread
        for(int tree_num = 0; tree_num < num_of_trees; ++tree_num){                                           //iterate over all trees
          __shared__ int thd_done_count;
          if (threadIdx.x == 0)
          {
            thd_done_count = 0;
          }
          unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];                                          //index into start of tree within subtree arrays
          unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;                      //number of subtrees for given tree
          int subtree_for_query = 0;                                                                          // Subtree for current query to traverse

          unsigned curr_subtree_idx = 0 ;                                                                     // Current subtree to traverse

          bool return_from_curr_tree = false;                                                                 //Indicates if we have reached a leaf in the tree

          while(curr_subtree_idx < num_of_subtrees){                                                          // Iterate over all subtrees of a tree
            if(thd_done_count == min(blockDim.x, queries_used)){                                              // If all active queries complete for this tree, break
              break;
            }
            int subtrees_loaded = 0;                                                                          // Number of subtrees loaded into shared mem
            while(subtrees_loaded < max_subtrees_loadable){                                                   // Iterate over subtrees that CAN be loaded into shared mem
             
              float *subtree_node_list = nodes + g_subtree_nodes_offset[tree_off_set+subtrees_loaded+curr_subtree_idx]*3 ; //Index into subtree node start
              int subtree_leaf_boundary = leaf_idx_boundry[tree_off_set+subtrees_loaded+curr_subtree_idx];    // Boundary of the subtree

              for (int node = threadIdx.x; node < subtree_leaf_boundary; node+=blockDim.x)                    // Assign each thread to a node to populate
              {
                subtree_space[(subtrees_loaded*nodes_per_subtree+node)*3] = subtree_node_list[node*3];        //feature_id
                subtree_space[(subtrees_loaded*nodes_per_subtree+node + 1)*3] = subtree_node_list[node*3+1];  //node_value
                subtree_space[(subtrees_loaded*nodes_per_subtree+node + 2)*3] = subtree_node_list[node*3+2];  //is_node_leaf
              }
              subtrees_loaded++;                                                                              // Subtree has been loaded, increment
            }

            __syncthreads();

            
            for(int st_traverse = curr_subtree_idx; st_traverse < curr_subtree_idx+subtrees_loaded; st_traverse++){ //Begin iterating over all loaded subtrees
              if(return_from_curr_tree){subtree_for_query = -2;break;}
              if(st_traverse == subtree_for_query){                                                           // If this subtree is one that the query should traverse...
                const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+st_traverse];         // Fetch leaf boundary of current subtree
      
                const unsigned *subtree_idx_to_subtree = idx_to_subtree + g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;
        
                
                
                //start from node 0
                unsigned curr_node = 0;

                while (true){
                  unsigned feature_id = subtree_space[((st_traverse-curr_subtree_idx)*nodes_per_subtree+curr_node)*3];
                  unsigned node_value = subtree_space[((st_traverse-curr_subtree_idx)*nodes_per_subtree+curr_node+1)*3];
                  unsigned is_tree_leaf = subtree_space[((st_traverse-curr_subtree_idx)*nodes_per_subtree+curr_node+2)*3];
                  
                  // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                  if (is_tree_leaf==1){ atomicAdd(results+tid, (unsigned)node_value); atomicInc(&thd_done_count,1);return_from_curr_tree = true; break; }
                  // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
                  bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
                  bool go_left = row[feature_id] <= node_value;
                  // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
                  if (not_subtree_bottom){
                      // go to left child in subtree
                      if (go_left)
                          curr_node = curr_node*2 + 1;
                      // go to right child in subtree
                      else
                          curr_node = curr_node*2 + 2;
                  // if reach bottom of subtree, then we need to go to another subtree
                  } else{
                      unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                      if (go_left)
                          subtree_for_query = subtree_idx_to_subtree[2*leaf_idx];
                      else
                          subtree_for_query = subtree_idx_to_subtree[2*leaf_idx+1];
                          
                      //stop the iterating of the current subtree, jump to the outer loop
                      //break;
                      break;
                  }

              //end subtree
                }
              }
            }
            curr_subtree_idx+=subtrees_loaded;
          }
          
        }
      }
      current_query+=queries_used;
      
}

#endif

#ifdef GPU_CSR 
__global__ void
csr_kernel(
  unsigned num_of_trees           ,
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

  unsigned num_of_queries         ,
  unsigned num_of_features        ,
  float *queries                  ,
  unsigned *results                  
){
    for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < num_of_queries; tid += blockDim.x*gridDim.x){
            //fetch a new query
            float * row = queries + tid*num_of_features; 
            //go over trees
            for(int i=0; i< num_of_trees; ++i){
                //csr layout
                //unsigned num_of_nodes = node_list_idx[i+1]-node_list_idx[i]-1;
                unsigned * node_list = node_list_total + node_list_idx[i];
                unsigned * edge_list = edge_list_total + edge_list_idx[i];
                unsigned * node_is_leaf = node_is_leaf_total + node_is_leaf_idx[i];
                unsigned * node_features = node_features_total + node_features_idx[i];
                float * node_values = node_values_total + node_values_idx[i];
        
                //start from node 0
                unsigned curr_node = 0;
                //iterate over nodes in a subtree
                while (true){
                    unsigned feature_id    = node_features[curr_node]; 
                    float node_value       = node_values[curr_node];
                    unsigned is_tree_leaf  = node_is_leaf[curr_node];
                    // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
                    if (is_tree_leaf==1){
                      //results[tid] = node_value;
//                        if (node_value == 1.0f){
                          atomicAdd(results+tid,(unsigned)node_value);
//                        }
                      break;
                    }
                    // if node is not leaf, we need two comparisons to decide if we keep traverse 
                    bool go_left = row[feature_id] <= node_value;
                    if (go_left)
                        curr_node = edge_list[node_list[curr_node]]; 
                    // go to right child in subtree
                    else
                        curr_node = edge_list[node_list[curr_node]+1]; 
                }
            }
    }     
}
#endif

__global__ void 
generate_results(unsigned num_of_queries, unsigned num_of_trees, unsigned * results){
    unsigned threshold = num_of_trees/2;
    for (int tid = blockDim.x*blockIdx.x + threadIdx.x; tid < num_of_queries; tid += blockDim.x*gridDim.x){
      if (results[tid] > threshold)
        results[tid] = 1;
      else
        results[tid] = 0;
    }
}


template <typename T>
unsigned read_arr(ifstream &infile, vector<T> &output ,string var_name);
template <typename T>
void read_2darr(ifstream &infile, vector<T> &output ,string var_name, unsigned &row, unsigned &cow);

float predict_tree_csr_layout(unsigned *node_list, unsigned *edge_list, unsigned *node_is_leaf, unsigned *node_features, float *node_values, float *row);

float predict_tree_gpu_layout(int num_of_trees, const unsigned *prefix_sum_subtree_nums, const float *nodes, const unsigned *idx_to_subtree, const unsigned *leaf_idx_boundry , const unsigned *g_subtree_nodes_offset, const unsigned *g_subtree_idx_to_subtree_offset, unsigned tree_num, float *row); 

int main(){

  cudaDeviceProp deviceProp;
  int numBlocks = 80;
  int threadsPerBlock = 64;

  //common data used by both csr and hier versions of GPU kernels
  ifstream infile;
  unsigned num_of_trees;
  INIT_TIMER 
  vector<unsigned> h_results;
  unsigned wrong_num = 0;
  unsigned long shared_mem_blk;

  cudaSetDevice(0);
  cudaGetDeviceProperties(&deviceProp, 0);

  shared_mem_blk = (float)SHARED_MEM_USAGE * deviceProp.sharedMemPerBlock;
  int subtree_max_depth = 5;
#ifdef GPU_HIER
  //read HIER data
  infile.open("treefile_hier.txt");
  string str;
  char   c;
  infile >> str;
  if (str!=string("num_of_trees")) {
    cout << str << "error reading num_of_trees";
  }
  infile >> num_of_trees >> c;
  cout << str << "\n" << num_of_trees << "\n";
  infile >> str;
  if (str!=string("prefix_sum_subtree_nums")) {
    cout << str << "error reading prefix_sum_subtree_nums";
  }
  unsigned len_prefix_sum_subtree_nums;
  infile >> len_prefix_sum_subtree_nums >> c;
  cout << str << "\n" << len_prefix_sum_subtree_nums << "\n";
  vector<unsigned> prefix_sum_subtree_nums(len_prefix_sum_subtree_nums,0);
  for (unsigned i = 0; i<len_prefix_sum_subtree_nums; ++i){
    infile >> prefix_sum_subtree_nums[i] >> c;
  }
  infile >> str;
  if (str!=string("nodes")) {
    cout << str << "error reading nodes";
  }
  unsigned len_nodes;
  infile >> len_nodes >> c;
  cout << str << "\n" << len_nodes << "\n";
  vector<float> nodes(len_nodes);
  for (unsigned i=0;i< len_nodes;++i){
    infile >> nodes[i] >> c;
  }
  infile >> str;
  if (str!=string("idx_to_subtree")) {
    cout << str << "error reading idx_to_subtree";
  }
  unsigned len_idx_to_subtree;
  infile >> len_idx_to_subtree >> c;
  cout << str << "\n" << len_idx_to_subtree << "\n";
  vector<unsigned> idx_to_subtree(len_idx_to_subtree,0);
  for (unsigned i = 0; i<len_idx_to_subtree; ++i){
    infile >> idx_to_subtree[i] >> c;
  }
  infile >> str;
  if (str!=string("leaf_idx_boundry")) {
    cout << str << "error reading leaf_idx_boundry";
  }
  unsigned len_leaf_idx_boundry;
  infile >> len_leaf_idx_boundry >> c;
  cout << str << "\n" << len_leaf_idx_boundry << "\n";
  vector<unsigned> leaf_idx_boundry(len_leaf_idx_boundry,0);
  for (unsigned i = 0; i<len_leaf_idx_boundry; ++i){
    infile >> leaf_idx_boundry[i] >> c;
  }
  infile >> str;
  if (str!=string("g_subtree_nodes_offset")) {
    cout << str << "error reading g_subtree_nodes_offset";
  }
  unsigned len_g_subtree_nodes_offset;
  infile >> len_g_subtree_nodes_offset >> c;
  cout << str << "\n" << len_g_subtree_nodes_offset << "\n";
  vector<unsigned> g_subtree_nodes_offset(len_g_subtree_nodes_offset,0);
  for (unsigned i = 0; i<len_g_subtree_nodes_offset; ++i){
    infile >> g_subtree_nodes_offset[i] >> c;
  }
  infile >> str;
  if (str!=string("g_subtree_idx_to_subtree_offset")) {
    cout << str << "error reading g_subtree_idx_to_subtree_offset";
  }
  unsigned len_g_subtree_idx_to_subtree_offset;
  infile >> len_g_subtree_idx_to_subtree_offset >> c;
  cout << str << "\n" << len_g_subtree_idx_to_subtree_offset << "\n";
  vector<unsigned> g_subtree_idx_to_subtree_offset(len_g_subtree_idx_to_subtree_offset,0);
  for (unsigned i = 0; i<len_g_subtree_idx_to_subtree_offset; ++i){
    float tmp;
    infile >> tmp >> c;
    g_subtree_idx_to_subtree_offset[i] = tmp; 
  }
  infile.close();
#endif

#ifdef GPU_CSR
  //read CSR data
  infile.open("treefile_csr.txt");
  vector<unsigned>   node_list_idx      ; 
  vector<unsigned>   edge_list_idx      ;
  vector<unsigned>   node_is_leaf_idx   ;
  vector<unsigned>   node_features_idx  ;
  vector<unsigned>   node_values_idx    ;
  vector<unsigned>   node_list_total    ;
  vector<unsigned>   edge_list_total    ;
  vector<unsigned>   node_is_leaf_total ;
  vector<unsigned>   node_features_total;
  vector<float>   node_values_total  ;
  num_of_trees = read_arr(infile, node_list_idx          , "node_list_idx"         );
  //read_arr returns size of array being read, size of node_list_idx = num_of_trees + 1
  num_of_trees = num_of_trees - 1;
  read_arr(infile, edge_list_idx          , "edge_list_idx"         );
  read_arr(infile, node_is_leaf_idx       , "node_is_leaf_idx"      );
  read_arr(infile, node_features_idx      , "node_features_idx"     );
  read_arr(infile, node_values_idx        , "node_values_idx"       );
  read_arr(infile, node_list_total        , "node_list_total"       );
  read_arr(infile, edge_list_total        , "edge_list_total"       );
  read_arr(infile, node_is_leaf_total     , "node_is_leaf_total"    );
  read_arr(infile, node_features_total    , "node_features_total"   );
  read_arr(infile, node_values_total      , "node_values_total"     );
  infile.close();
#endif
 

 //read input data 
  infile.open("./tree_input.txt");
  vector<float> X_test;
  vector<float> y_test;
  unsigned row,col;
  read_2darr(infile, X_test, "X_test", row, col);
  cout << "X_test" << " with " << row << " rows" << " and " << col << " cols.\n";
  read_arr(infile, y_test, "y_test");
  infile.close();

//NOW we copy input and allocate output to/on GPU
  float *d_queries;
  cudaMalloc((void**)&d_queries, sizeof(float)*row*col);

  unsigned *d_results;
  cudaMalloc((void**)&d_results, sizeof(unsigned)*row);

  cudaMemcpy(d_queries, X_test.data(), sizeof(float)*row*col, cudaMemcpyHostToDevice);

#ifdef GPU_CSR
  cout << "Allocating csr data on GPU" << endl;

//NOW we have and need to copy these consolidated csr format to GPU 
  cout << "Copying csr data to GPU" << endl;
  // unsigned num_of_trees;
  //vector<unsigned>   node_list_idx      ; 
  //vector<unsigned>   edge_list_idx      ;
  //vector<unsigned>   node_is_leaf_idx   ;
  //vector<unsigned>   node_features_idx  ;
  //vector<unsigned>   node_values_idx    ;

  //vector<unsigned>   node_list_total    ;
  //vector<unsigned>   edge_list_total    ;
  //vector<unsigned>   node_is_leaf_total ;
  //vector<unsigned>   node_features_total;
  //vector<float>   node_values_total  ;

  unsigned *   d_node_list_idx      ; 
  unsigned *   d_edge_list_idx      ;
  unsigned *   d_node_is_leaf_idx   ;
  unsigned *   d_node_features_idx  ;
  unsigned *   d_node_values_idx    ;

  unsigned *   d_node_list_total    ;
  unsigned *   d_edge_list_total    ;
  unsigned *   d_node_is_leaf_total ;
  unsigned *   d_node_features_total;
  float    *   d_node_values_total  ;


  cudaMalloc((void **) &d_node_list_idx, sizeof(unsigned)*node_list_idx.size());
  cudaMalloc((void **) &d_edge_list_idx, sizeof(unsigned)*edge_list_idx.size());
  cudaMalloc((void **) &d_node_is_leaf_idx, sizeof(unsigned)*node_is_leaf_idx.size());
  cudaMalloc((void **) &d_node_features_idx, sizeof(unsigned)*node_features_idx.size());
  cudaMalloc((void **) &d_node_values_idx, sizeof(unsigned)*node_values_idx.size());
  
  cudaMalloc((void **) &d_node_list_total, sizeof(unsigned)*node_list_total.size());
  cudaMalloc((void **) &d_edge_list_total, sizeof(unsigned)*edge_list_total.size());
  cudaMalloc((void **) &d_node_is_leaf_total, sizeof(unsigned)*node_is_leaf_total.size());
  cudaMalloc((void **) &d_node_features_total, sizeof(unsigned)*node_features_total.size());
  cudaMalloc((void **) &d_node_values_total, sizeof(unsigned)*node_values_total.size());

  cudaMemcpy(d_node_list_idx,              node_list_idx.data()    ,         sizeof(unsigned)*node_list_idx.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_edge_list_idx,              edge_list_idx.data()    ,         sizeof(unsigned)*edge_list_idx.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_is_leaf_idx,        node_is_leaf_idx.data()    ,      sizeof(unsigned)*node_is_leaf_idx.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_features_idx,      node_features_idx.data()    ,     sizeof(unsigned)*node_features_idx.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_values_idx,          node_values_idx.data()    ,       sizeof(unsigned)*node_values_idx.size(),cudaMemcpyHostToDevice );

  cudaMemcpy(d_node_list_total,          node_list_total.data()    ,       sizeof(unsigned)*node_list_total.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_edge_list_total,          edge_list_total.data()    ,       sizeof(unsigned)*edge_list_total.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_is_leaf_total,    node_is_leaf_total.data()    ,    sizeof(unsigned)*node_is_leaf_total.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_features_total,  node_features_total.data()    ,   sizeof(unsigned)*node_features_total.size(),cudaMemcpyHostToDevice );
  cudaMemcpy(d_node_values_total,      node_values_total.data()    ,     sizeof(unsigned)*node_values_total.size(),cudaMemcpyHostToDevice );


  cout << "Start executing csr format on GPU" << endl;
  cudaMemset(d_results, 0 , row*sizeof(unsigned));
  cout << cudaGetErrorName(cudaGetLastError()) << endl;
  START_TIMER
  csr_kernel<<<80,256>>>(
                          num_of_trees           ,
                          d_node_list_idx      ,
                          d_edge_list_idx      ,
                          d_node_is_leaf_idx   ,
                          d_node_features_idx  ,
                          d_node_values_idx    ,

                          d_node_list_total    ,
                          d_edge_list_total    ,
                          d_node_is_leaf_total ,
                          d_node_features_total,
                          d_node_values_total  ,

                          row                  ,
                          col                  ,
                          d_queries            ,
                          d_results                  
  );
  generate_results<<<80,256>>>(row, num_of_trees, d_results);
  cudaDeviceSynchronize();
  STOP_TIMER("csr kernel")
  cout << "Kernel returned:" << cudaGetErrorName(cudaGetLastError()) << endl;
  h_results.resize(row);
  cudaMemcpy( h_results.data(), d_results, sizeof(unsigned)*row, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  wrong_num = 0;
  for(auto i=0; i < row;++i){
    if (h_results[i]!=y_test[i]){
      wrong_num++;
    }
  }
  cout << "csr result is wrong with this many: " << wrong_num << endl;
  cout << "accuracy rate: " << (float)(row-wrong_num)/(float)row << endl;

  //destroy csr related data on GPU
  cudaFree(d_node_list_idx)      ;             
  cudaFree(d_edge_list_idx)      ;    
  cudaFree(d_node_is_leaf_idx)   ;    
  cudaFree(d_node_features_idx)  ;   
  cudaFree(d_node_values_idx)    ;   
  cudaFree(d_node_list_total)    ;    
  cudaFree(d_edge_list_total)    ;   
  cudaFree(d_node_is_leaf_total) ;       
  cudaFree(d_node_features_total);   
  cudaFree(d_node_values_total)  ; 
#endif
  
#ifdef GPU_HIER
/*
//validate GPU hierarchcal layout on CPU
  cout << "Test csr and hier results with " << row << " test samples.\n";
  for(int query = 0; query < row; ++query){
    float* row = X_test.data() + query*col;
    for(int i = 0; i < num_of_trees; ++i){
      //hier layout
      float res1 = predict_tree_gpu_layout(num_of_trees, prefix_sum_subtree_nums.data(), nodes.data(), idx_to_subtree.data(), leaf_idx_boundry.data() ,g_subtree_nodes_offset.data(), g_subtree_idx_to_subtree_offset.data(), i , row); 
      //cout << "Result1:" << res1 << "\n";

      //csr layout
      unsigned num_of_nodes = node_list_idx[i+1]-node_list_idx[i]-1;
      unsigned * node_list = node_list_total.data() + node_list_idx[i];
      unsigned * edge_list = edge_list_total.data() + edge_list_idx[i];
      unsigned * node_is_leaf = node_is_leaf_total.data() + node_is_leaf_idx[i];
      unsigned * node_features = node_features_total.data() + node_features_idx[i];
      float * node_values = node_values_total.data() + node_values_idx[i];
      
      float res2 = predict_tree_csr_layout(node_list, edge_list, node_is_leaf, node_features, node_values, row);
      //cout << "Result2:" << res2 << "\n";
      if (res1!=res2){
        cout << "csr and hier results don't match" << res1 << " res1|res2 "<< res2 << "\n";
      }
    }
  }
  cout << "csr and hier results match, verification passes" << endl;
*/


//NOW we copy these hier tree format to GPU 
//       unsigned num_of_trees                              ;
//       vector<unsigned> prefix_sum_subtree_nums           ;
//       vector<float   > nodes                             ;
//       vector<unsigned> idx_to_subtree                    ;
//       vector<unsigned> leaf_idx_boundry                  ;
//       vector<unsigned> g_subtree_nodes_offset            ;
//       vector<unsigned> g_subtree_idx_to_subtree_offset   ;

  cout << "Allocating hier format on GPU" << endl;
  unsigned *d_prefix_sum_subtree_nums        ;   
  float    *d_nodes                          ;
  unsigned *d_idx_to_subtree                 ;
  unsigned *d_leaf_idx_boundry               ;
  unsigned *d_g_subtree_nodes_offset         ;
  unsigned *d_g_subtree_idx_to_subtree_offset;
  
  cudaMalloc((void**)&d_prefix_sum_subtree_nums              ,sizeof( unsigned )*prefix_sum_subtree_nums.size()            );
  cudaMalloc((void**)&d_nodes                                ,sizeof( float    )*nodes.size()                              );
  cudaMalloc((void**)&d_idx_to_subtree                       ,sizeof( unsigned )*idx_to_subtree.size()                     );
  cudaMalloc((void**)&d_leaf_idx_boundry                     ,sizeof( unsigned )*leaf_idx_boundry.size()                   );
  cudaMalloc((void**)&d_g_subtree_nodes_offset               ,sizeof( unsigned )*g_subtree_nodes_offset.size()             );
  cudaMalloc((void**)&d_g_subtree_idx_to_subtree_offset      ,sizeof( unsigned )*g_subtree_idx_to_subtree_offset.size()    );
  
  cout << "Copying hier format to GPU" << endl;
  cudaMemcpy( d_prefix_sum_subtree_nums        ,prefix_sum_subtree_nums.data()        ,sizeof( unsigned )*prefix_sum_subtree_nums.size()        ,cudaMemcpyHostToDevice);
  cudaMemcpy( d_nodes                          ,nodes.data()                          ,sizeof( float    )*nodes.size()                          ,cudaMemcpyHostToDevice);
  cudaMemcpy( d_idx_to_subtree                 ,idx_to_subtree.data()                 ,sizeof( unsigned )*idx_to_subtree.size()                 ,cudaMemcpyHostToDevice);
  cudaMemcpy( d_leaf_idx_boundry               ,leaf_idx_boundry.data()               ,sizeof( unsigned )*leaf_idx_boundry.size()               ,cudaMemcpyHostToDevice);
  cudaMemcpy( d_g_subtree_nodes_offset         ,g_subtree_nodes_offset.data()         ,sizeof( unsigned )*g_subtree_nodes_offset.size()         ,cudaMemcpyHostToDevice);
  cudaMemcpy( d_g_subtree_idx_to_subtree_offset,g_subtree_idx_to_subtree_offset.data(),sizeof( unsigned )*g_subtree_idx_to_subtree_offset.size(),cudaMemcpyHostToDevice);
  
  cout << "Start executing hier format on GPU" << endl;
  cout << cudaGetErrorName(cudaGetLastError()) << endl;
  //reset result array to 0
  cudaMemset(d_results, 0 , row*sizeof(unsigned));

  // unsigned long sm_bytes_available,
  // float ratio_queries             ,
  // int batch_start                 ,
  // int batch_end                   ,
  // int max_st_depth

  START_TIMER
  hier_kernel<<<numBlocks,threadsPerBlock, shared_mem_blk>>>(
                          num_of_trees                     ,
                          d_prefix_sum_subtree_nums        ,
                          d_nodes                          ,
                          d_idx_to_subtree                 ,  
                          d_leaf_idx_boundry               ,
                          d_g_subtree_nodes_offset         ,  
                          d_g_subtree_idx_to_subtree_offset,  
  
                          row                              ,
                          col                              ,
                          d_queries                        ,
                          d_results                        ,
                          shared_mem_blk-4                 ,
                          RATIO_QUERIES                    ,
                          subtree_max_depth                ,
                          row
  );
  generate_results<<<numBlocks,threadsPerBlock>>>(row, num_of_trees, d_results);
  cudaDeviceSynchronize();
  STOP_TIMER("hier kernel")
  cout << "Kernel returned:" << cudaGetErrorName(cudaGetLastError()) << endl;

  h_results.resize(row);
  cudaMemcpy( h_results.data(), d_results, sizeof(unsigned)*row, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  wrong_num = 0;
  for(auto i=0; i < row;++i){
    if (h_results[i]!=y_test[i]){
      wrong_num++;
    }
  }
  cout << "hier result is wrong with this many: " << wrong_num << endl;
  cout << "accuracy rate: " << (float)(row-wrong_num)/(float)row << endl;
#endif
  //main returns
  return 0;
}

//predict the result over a decision_tree
float predict_tree_gpu_layout(int num_of_trees, const unsigned *prefix_sum_subtree_nums, const float *nodes, const unsigned *idx_to_subtree, const unsigned *leaf_idx_boundry , const unsigned *g_subtree_nodes_offset, const unsigned *g_subtree_idx_to_subtree_offset, unsigned tree_num, float *row) {

    unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
    //unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;

    unsigned curr_subtree_idx = 0 ;  
    //iterate over subtree
    while (true){
        //fetch the subtree nodes
        const float *subtree_node_list;
        subtree_node_list = nodes + g_subtree_nodes_offset[tree_off_set+curr_subtree_idx]*3 ;

        //fetch subtree_leaf_idx_boundry
        const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set+curr_subtree_idx];

        //fetch subtree_idx_to_other_subtree
        const unsigned *subtree_idx_to_subtree = idx_to_subtree + g_subtree_idx_to_subtree_offset[tree_off_set+curr_subtree_idx]*2;

        //start from node 0
        unsigned curr_node = 0;
        //iterate over nodes in a subtree
        while (true){
            unsigned feature_id = subtree_node_list[curr_node*3];
            float node_value    = subtree_node_list[curr_node*3+1];
            unsigned is_tree_leaf    = subtree_node_list[curr_node*3+2];

            // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
            if (is_tree_leaf==1)
                return node_value;

            // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
            bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
            bool go_left = row[feature_id] <= node_value;

            // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
            if (not_subtree_bottom){
                // go to left child in subtree
                if (go_left)
                    curr_node = curr_node*2 + 1;
                // go to right child in subtree
                else
                    curr_node = curr_node*2 + 2;
            // if reach bottom of subtree, then we need to go to another subtree
            } else{
                unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                if (go_left)
                    curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx];
                else
                    curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx+1];
                //stop the iterating of the current subtree, jump to the outer loop
                break;
            }
        }
    }
}

template <typename T>
unsigned read_arr(ifstream &infile, vector<T> &output, string var_name){
  string str;
  char c;
  infile >> str;
  if (str!=var_name) {
    cout << str << "error reading " << var_name << endl;
  }
  unsigned len;
  infile >> len >> c;
  output.resize(len);
  for (unsigned i = 0; i<len; ++i){
    infile >> output[i] >> c;
  }
//  cout << "Read " << str << " with " << len << " elements\n";
  return len;
}

template <typename T>
void read_2darr(ifstream &infile, vector<T> &output, string var_name, unsigned &row, unsigned &col){
  string str;
  char c;
  infile >> str;
  if (str!=var_name) {
    cout << str << "error reading " << var_name << endl;
  }
  unsigned nrow,ncol;
  infile >> nrow >> c >> ncol >> c;
  row = nrow;
  col = ncol;
  output.resize(nrow*ncol);
  for (unsigned i = 0; i<nrow*ncol; ++i){
    infile >> output[i] >> c;
  }
//  cout << "Read " << str << " with " << nrow << " rows" << " and " << ncol << " cols.\n";
}

//    node_list = tree[1]
//    edge_list = tree[2]
//    node_is_leaf = tree[3]
//    node_features = tree[4]
//    node_values = tree[5]
float predict_tree_csr_layout(unsigned *node_list, unsigned *edge_list, unsigned *node_is_leaf, unsigned *node_features, float *node_values, float *row){
    //start from node 0
    unsigned curr_node = 0;
    //iterate over nodes in a subtree
    while (true){
        unsigned feature_id    = node_features[curr_node]; 
        float node_value       = node_values[curr_node];
        unsigned is_tree_leaf  = node_is_leaf[curr_node];
        // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
        if (is_tree_leaf==1)
            return node_value;
        // if node is not leaf, we need two comparisons to decide if we keep traverse 
        bool go_left = row[feature_id] <= node_value;
        if (go_left)
            curr_node = edge_list[node_list[curr_node]]; 
        // go to right child in subtree
        else
            curr_node = edge_list[node_list[curr_node]+1]; 
    }
}




//Alternative way to traverse a node
//                    if (go_left){
//                        if(not_subtree_bottom){
//                          curr_node = curr_node*2 + 1;
//                        }else{
//                          unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
//                          curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx];
//                          break;
//                        }
//                    // go to right child in subtree
//                    } else {
//                        if(not_subtree_bottom){
//                          curr_node = curr_node*2 + 2;
//                        }else{
//                          unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
//                          curr_subtree_idx = subtree_idx_to_subtree[2*leaf_idx+1];
//                          break;
//                        }
//                    }
