#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>
#include "xcl2.hpp"

#define TIMING

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name) std::cout << "RUNTIME of " << name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

#ifdef BATCH
typedef struct {
   unsigned short query_results;
   unsigned int query_curr_subtree;
   unsigned int query_data_buf;
} query_t;
#endif


template <typename T>
unsigned read_arr(std::ifstream &infile, std::vector<T, aligned_allocator<T>> &output, std::string var_name);
template <typename T>
void read_2darr(std::ifstream &infile, std::vector<T, aligned_allocator<T>> &output, std::string var_name, unsigned &row, unsigned &cow);

float predict_tree_csr_layout(unsigned *node_list, unsigned *edge_list, unsigned *node_is_leaf, unsigned *node_features, float *node_values, float *row);

float predict_tree_fpga_layout(int num_of_trees, const unsigned *prefix_sum_subtree_nums, const float *nodes, const unsigned *idx_to_subtree, const unsigned *leaf_idx_boundry, const unsigned *g_subtree_nodes_offset, const unsigned *g_subtree_idx_to_subtree_offset, unsigned tree_num, float *row);

int main(int argc, char** argv)
{

    std::string binaryFile = argv[1];
    std::string treeFile = argv[2];
    std::string inputFile = argv[3];

    // common data used by both csr and hier versions of FPGA kernels
    std::ifstream infile;
    unsigned num_of_trees;
    INIT_TIMER
    unsigned wrong_num = 0;
    cl_int err;
    cl::CommandQueue q;
    std::vector<cl::Kernel> rf_kernel(NUM_SLRS*NUM_CUS);
    #if SPLIT
        std::vector<cl::Kernel> rf_kernel_burst(NUM_SLRS);
    #endif
    // cl::Kernel generate_results_kernel;
    cl::Context context;

    auto devices = xcl::get_xil_devices();

    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};


    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++)
    {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            std::cout << "Device[" << i << "]: program successful!\n";
            // This call will extract a kernel out of the program we loaded in the
            // previous line. A kernel is an OpenCL function that is executed on the
            // FPGA. This function is defined in the src/vetor_addition.cl file.
            for(int i = 0; i < NUM_SLRS * NUM_CUS; i++){
                std::string strnum = std::to_string(i);
                #ifdef FPGA_CSR
                std::string base_name = "csr_kernel:{csr_kernel_";
                std::string total_name = base_name + strnum + "}";
                std::cout << "Retrieving " << total_name << std::endl;
                if(NUM_SLRS == 1) total_name = "csr_kernel";
                #endif
                #ifdef FPGA_HIER
                std::string base_name = "hier_kernel:{hier_kernel_";
                std::string total_name = base_name + strnum + "}";
                std::cout << "Retrieving " << total_name << std::endl;
                if(NUM_SLRS == 1) total_name = "hier_kernel";
                #endif
                cl::Kernel temp;
                OCL_CHECK(err, temp = cl::Kernel(program, total_name.c_str(), &err));
                rf_kernel[i] = temp;
            }
            #if SPLIT
                for(int i = 0; i < NUM_SLRS; i++){
                std::string strnum = std::to_string(i);
                #ifdef FPGA_CSR
                std::string base_name = "csr_kernel_burst:{csr_kernel_burst_";
                std::string total_name = base_name + strnum + "}";
                std::cout << "Retrieving " << total_name << std::endl;
                if(NUM_SLRS == 1) total_name = "csr_kernel_burst_";
                #endif
                #ifdef FPGA_HIER
                std::string base_name = "hier_kernel_burst:{hier_kernel_burst_";
                std::string total_name = base_name + strnum + "}";
                std::cout << "Retrieving " << total_name << std::endl;
                if(NUM_SLRS == 1) total_name = "hier_kernel_burst";
                #endif
                cl::Kernel temp;
                OCL_CHECK(err, temp = cl::Kernel(program, total_name.c_str(), &err));
                rf_kernel_burst[i] = temp;
            }
            #endif
            // OCL_CHECK(err, generate_results_kernel = cl::Kernel(program, "generate_results", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

#ifdef FPGA_HIER
    // read HIER data
    infile.open(treeFile);
    std::string str;
    char c;
    infile >> str;
    if (str != std::string("num_of_trees"))
    {
        std::cout << str << "error reading num_of_trees";
    }
    infile >> num_of_trees >> c;
    std::cout << str << "\n"
         << num_of_trees << "\n";
    infile >> str;
    if (str != std::string("prefix_sum_subtree_nums"))
    {
        std::cout << str << "error reading prefix_sum_subtree_nums";
    }
    unsigned len_prefix_sum_subtree_nums;
    infile >> len_prefix_sum_subtree_nums >> c;
    std::cout << str << "\n"
         << len_prefix_sum_subtree_nums << "\n";
    std::vector<unsigned, aligned_allocator<unsigned>> prefix_sum_subtree_nums(len_prefix_sum_subtree_nums, 0);
    for (unsigned i = 0; i < len_prefix_sum_subtree_nums; ++i)
    {
        infile >> prefix_sum_subtree_nums[i] >> c;
    }
    infile >> str;
    if (str != std::string("nodes"))
    {
        std::cout << str << "error reading nodes";
    }
    unsigned len_nodes;
    infile >> len_nodes >> c;
    std::cout << str << "\n"
         << len_nodes << "\n";
    std::vector<float, aligned_allocator<float> > nodes(len_nodes);
    for (unsigned i = 0; i < len_nodes; ++i)
    {
        infile >> nodes[i] >> c;
    }
    infile >> str;
    if (str != std::string("idx_to_subtree"))
    {
        std::cout << str << "error reading idx_to_subtree";
    }
    unsigned len_idx_to_subtree;
    infile >> len_idx_to_subtree >> c;
    std::cout << str << "\n"
         << len_idx_to_subtree << "\n";
    std::vector<unsigned, aligned_allocator<unsigned> > idx_to_subtree(len_idx_to_subtree, 0);
    for (unsigned i = 0; i < len_idx_to_subtree; ++i)
    {
        infile >> idx_to_subtree[i] >> c;
    }
    infile >> str;
    if (str != std::string("leaf_idx_boundry"))
    {
        std::cout << str << "error reading leaf_idx_boundry";
    }
    unsigned len_leaf_idx_boundry;
    infile >> len_leaf_idx_boundry >> c;
    std::cout << str << "\n"
         << len_leaf_idx_boundry << "\n";
    std::vector<unsigned, aligned_allocator<unsigned> > leaf_idx_boundry(len_leaf_idx_boundry, 0);
    for (unsigned i = 0; i < len_leaf_idx_boundry; ++i)
    {
        infile >> leaf_idx_boundry[i] >> c;
    }
    infile >> str;
    if (str != std::string("g_subtree_nodes_offset"))
    {
        std::cout << str << "error reading g_subtree_nodes_offset";
    }
    unsigned len_g_subtree_nodes_offset;
    infile >> len_g_subtree_nodes_offset >> c;
    std::cout << str << "\n"
         << len_g_subtree_nodes_offset << "\n";
    std::vector<unsigned, aligned_allocator<unsigned> > g_subtree_nodes_offset(len_g_subtree_nodes_offset, 0);
    for (unsigned i = 0; i < len_g_subtree_nodes_offset; ++i)
    {
        infile >> g_subtree_nodes_offset[i] >> c;
    }
    infile >> str;
    if (str != std::string("g_subtree_idx_to_subtree_offset"))
    {
        std::cout << str << "error reading g_subtree_idx_to_subtree_offset";
    }
    unsigned len_g_subtree_idx_to_subtree_offset;
    infile >> len_g_subtree_idx_to_subtree_offset >> c;
    std::cout << str << "\n"
         << len_g_subtree_idx_to_subtree_offset << "\n";
    std::vector<unsigned, aligned_allocator<unsigned> > g_subtree_idx_to_subtree_offset(len_g_subtree_idx_to_subtree_offset, 0);
    for (unsigned i = 0; i < len_g_subtree_idx_to_subtree_offset; ++i)
    {
        float tmp;
        infile >> tmp >> c;
        g_subtree_idx_to_subtree_offset[i] = tmp;
    }
    infile.close();
#endif

#ifdef FPGA_CSR
  //read CSR data
  infile.open(treeFile);
  std::vector<unsigned, aligned_allocator<unsigned>>   node_list_idx      ; 
  std::vector<unsigned, aligned_allocator<unsigned>>   edge_list_idx      ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_is_leaf_idx   ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_features_idx  ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_values_idx    ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_list_total    ;
  std::vector<unsigned, aligned_allocator<unsigned>>   edge_list_total    ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_is_leaf_total ;
  std::vector<unsigned, aligned_allocator<unsigned>>   node_features_total;
  std::vector<float, aligned_allocator<float>>   node_values_total  ;
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

    // read input data
    infile.open(inputFile);
    std::vector<float, aligned_allocator<float> > X_test;
    std::vector<float, aligned_allocator<float> > y_test;
    unsigned row, col;
    read_2darr(infile, X_test, "X_test", row, col);
    std::cout << "X_test"
         << " with " << row << " rows"
         << " and " << col << " cols.\n";
    read_arr(infile, y_test, "y_test");
    infile.close();

    std::vector<std::vector<unsigned, aligned_allocator<unsigned> >> h_results;

    for(int i = 0; i < NUM_SLRS; i++){
        size_t size;
        if(i < NUM_SLRS-1) size = row/NUM_SLRS;
        else size = (row / NUM_SLRS) + (row % NUM_SLRS);
        std::vector<unsigned, aligned_allocator<unsigned> > result(size, 0);
        h_results.push_back(result);
    }

    std::vector<float, aligned_allocator<float>> X_partitioned[NUM_SLRS];

    for (int i = 0; i < NUM_SLRS; ++i)
    {
        size_t size = (row/NUM_SLRS) * col;
        size_t end_size;
        if(i < NUM_SLRS - 1){
            end_size = size;
        }
        else{
            end_size = size + (row % NUM_SLRS) * col;
        }
        // get range for the next set of `n` elements
        auto start_itr = std::next(X_test.cbegin(), i*size);
        auto end_itr = std::next(X_test.cbegin(), i*size + end_size);
 
        // allocate memory for the sub-vector
        X_partitioned[i].resize(end_size);
 
        // copy elements from the input range to the sub-vector
        std::copy(start_itr, end_itr, X_partitioned[i].begin());
        std::cout << "X_partition[" << i << "] size = " << X_partitioned[i].size() << std::endl;
    }

    // NOW we allocate input and output to/on FPGA

    cl_mem_ext_ptr_t queries_alloc[NUM_SLRS];
    std::vector<cl::Buffer> d_queries;

    for(int i = 0; i < NUM_SLRS; i++){
        unsigned num_queries;
        if(i < NUM_SLRS-1) num_queries = row/NUM_SLRS;
        else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
        queries_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        queries_alloc[i].param = 0;
        queries_alloc[i].obj   = X_partitioned[i].data();
        OCL_CHECK(err, cl::Buffer d_queries_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(float) * num_queries * col, &queries_alloc[i], &err));
        d_queries.push_back(d_queries_temp);
    }

    cl_mem_ext_ptr_t results_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_results;

    for(int i = 0; i < NUM_SLRS; i++){
        results_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        results_alloc[i].param = 0;
        results_alloc[i].obj   = h_results[i].data();
        OCL_CHECK(err, cl::Buffer d_results_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sizeof(float) * h_results[i].size(), &results_alloc[i], &err));
        d_results.push_back(d_results_temp);
    }

    // OCL_CHECK(err, cl::Buffer d_queries(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float) * row * col, X_test.data(), &err));
    // OCL_CHECK(err, cl::Buffer d_results(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(unsigned) * row, h_results.data(), &err));

#ifdef FPGA_CSR
  std::cout << "Allocating csr data on FPGA" << std::endl;

//NOW we have and need to copy these consolidated csr format to GPU 
  std::cout << "Copying csr data to FPGA" << std::endl;
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

  cl_mem_ext_ptr_t d_node_list_idx_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_list_idx;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_list_idx_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_list_idx_alloc[i].param = 0;
      d_node_list_idx_alloc[i].obj = node_list_idx.data();
      OCL_CHECK(err, cl::Buffer d_node_list_idx_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_list_idx.size(), &d_node_list_idx_alloc[i], &err));
      d_node_list_idx.push_back(d_node_list_idx_temp);
  }
  cl_mem_ext_ptr_t d_edge_list_idx_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_edge_list_idx;
  for(int i = 0; i < NUM_SLRS; i++){
      d_edge_list_idx_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_edge_list_idx_alloc[i].param = 0;
      d_edge_list_idx_alloc[i].obj = edge_list_idx.data();
      OCL_CHECK(err, cl::Buffer d_edge_list_idx_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * edge_list_idx.size(), &d_edge_list_idx_alloc[i], &err));
      d_edge_list_idx.push_back(d_edge_list_idx_temp);
  }
  cl_mem_ext_ptr_t d_node_is_leaf_idx_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_is_leaf_idx;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_is_leaf_idx_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_is_leaf_idx_alloc[i].param = 0;
      d_node_is_leaf_idx_alloc[i].obj = node_is_leaf_idx.data();
      OCL_CHECK(err, cl::Buffer d_node_is_leaf_idx_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_is_leaf_idx.size(), &d_node_is_leaf_idx_alloc[i], &err));
      d_node_is_leaf_idx.push_back(d_node_is_leaf_idx_temp);
  }
  cl_mem_ext_ptr_t d_node_features_idx_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_features_idx;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_features_idx_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_features_idx_alloc[i].param = 0;
      d_node_features_idx_alloc[i].obj = node_features_idx.data();
      OCL_CHECK(err, cl::Buffer d_node_features_idx_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_features_idx.size(), &d_node_features_idx_alloc[i], &err));
      d_node_features_idx.push_back(d_node_features_idx_temp);
  }
  cl_mem_ext_ptr_t d_node_values_idx_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_values_idx;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_values_idx_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_values_idx_alloc[i].param = 0;
      d_node_values_idx_alloc[i].obj = node_values_idx.data();
      OCL_CHECK(err, cl::Buffer d_node_values_idx_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_values_idx.size(), &d_node_values_idx_alloc[i], &err));
      d_node_values_idx.push_back(d_node_values_idx_temp);
  }

  cl_mem_ext_ptr_t d_node_list_total_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_list_total;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_list_total_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_list_total_alloc[i].param = 0;
      d_node_list_total_alloc[i].obj = node_list_total.data();
      OCL_CHECK(err, cl::Buffer d_node_list_total_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_list_total.size(), &d_node_list_total_alloc[i], &err));
      d_node_list_total.push_back(d_node_list_total_temp);
  }
  cl_mem_ext_ptr_t d_edge_list_total_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_edge_list_total;
  for(int i = 0; i < NUM_SLRS; i++){
      d_edge_list_total_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_edge_list_total_alloc[i].param = 0;
      d_edge_list_total_alloc[i].obj = edge_list_total.data();
      OCL_CHECK(err, cl::Buffer d_edge_list_total_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * edge_list_total.size(), &d_edge_list_total_alloc[i], &err));
      d_edge_list_total.push_back(d_edge_list_total_temp);
  }
  cl_mem_ext_ptr_t d_node_is_leaf_total_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_is_leaf_total;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_is_leaf_total_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_is_leaf_total_alloc[i].param = 0;
      d_node_is_leaf_total_alloc[i].obj = node_is_leaf_total.data();
      OCL_CHECK(err, cl::Buffer d_node_is_leaf_total_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_is_leaf_total.size(), &d_node_is_leaf_total_alloc[i], &err));
      d_node_is_leaf_total.push_back(d_node_is_leaf_total_temp);
  }
  cl_mem_ext_ptr_t d_node_features_total_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_features_total;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_features_total_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_features_total_alloc[i].param = 0;
      d_node_features_total_alloc[i].obj = node_features_total.data();
      OCL_CHECK(err, cl::Buffer d_node_features_total_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * node_features_total.size(), &d_node_features_total_alloc[i], &err));
      d_node_features_total.push_back(d_node_features_total_temp);
  }
  cl_mem_ext_ptr_t d_node_values_total_alloc[NUM_SLRS];
  std::vector<cl::Buffer> d_node_values_total;
  for(int i = 0; i < NUM_SLRS; i++){
      d_node_values_total_alloc[i].flags = i | XCL_MEM_TOPOLOGY;
      d_node_values_total_alloc[i].param = 0;
      d_node_values_total_alloc[i].obj = node_values_total.data();
      OCL_CHECK(err, cl::Buffer d_node_values_total_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(float) * node_values_total.size(), &d_node_values_total_alloc[i], &err));
      d_node_values_total.push_back(d_node_values_total_temp);
  }

  std::vector<cl::Event> write_event(NUM_SLRS);

  for(int i = 0; i < NUM_SLRS; i++){
    unsigned num_queries;
    if(i < NUM_SLRS-1) num_queries = row/NUM_SLRS;
    else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
    unsigned start = 0;
    int narg = 0;
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_of_trees));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_list_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_edge_list_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_is_leaf_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_features_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_values_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_list_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_edge_list_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_is_leaf_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_features_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_values_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, start));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_queries));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, col));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_queries[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_results[i]));

    std::cout << "Copying csr format to FPGA SLR " << i << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_node_list_idx[i],
                                                     d_edge_list_idx[i],
                                                     d_node_is_leaf_idx[i],
                                                     d_node_features_idx[i],
                                                     d_node_values_idx[i],
                                                     d_node_list_total[i],
                                                     d_edge_list_total[i],
                                                     d_node_is_leaf_total[i],
                                                     d_node_features_total[i],
                                                     d_node_values_total[i],
                                                     d_queries[i],
                                                     d_results[i]}, 0 /* 0 means from host*/,
                                                    nullptr, &write_event[i]));
}

std::vector<cl::Event> rf_event(NUM_SLRS*NUM_CUS);

for(int i = 0; i < NUM_SLRS; i++){
    int narg = 0; 
    unsigned num_queries;
    if(i < (NUM_SLRS-1)*NUM_CUS) num_queries = row/NUM_SLRS;
    else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
    unsigned num_queries_per_cu = num_queries / NUM_CUS;
    unsigned start = num_queries_per_cu * (i % NUM_CUS);
    unsigned end;
    if(i % NUM_CUS == NUM_CUS-1){
        end = num_queries;
    }
    else{
        end = num_queries_per_cu * ((i % NUM_CUS) + 1);
    }
    std::cout << i << " CU start is " << start << ", end is " << end << std::endl;
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_of_trees));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_list_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_edge_list_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_is_leaf_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_features_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_values_idx[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_list_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_edge_list_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_is_leaf_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_features_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_node_values_total[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, start));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_queries));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, col));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_queries[i]));
    OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_results[i]));
}
START_TIMER
for(int i = 0; i < NUM_SLRS * NUM_CUS; i++){
    std::cout << "Start executing csr format on FPGA SLR" << i << std::endl;
    OCL_CHECK(err, err = q.enqueueTask(rf_kernel[i], &write_event, &rf_event[i]));
}


std::vector<cl::Event> read_event(NUM_SLRS);

for(int i = 0; i < NUM_SLRS; i++){
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_results[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &rf_event,
                                                    &read_event[i]));
}

OCL_CHECK(err, err = cl::Event::waitForEvents(read_event));
STOP_TIMER("csr kernel")

  wrong_num = 0;
    for (size_t i = 0; i < row; ++i)
    {
        size_t results_array = i/(row/NUM_SLRS);
        if(results_array >= NUM_SLRS) results_array = NUM_SLRS - 1;
        size_t offset = results_array * (row/NUM_SLRS);
        if (h_results[results_array][i - offset] != y_test[i])
        {
            wrong_num++;
        }
    }
  std::cout << "csr result is wrong with this many: " << wrong_num << std::endl;
  std::cout << "accuracy rate: " << (float)(row-wrong_num)/(float)row << std::endl;

#endif

#ifdef FPGA_HIER

    // Mod nodes layout
    // We divide nodes array into two arrays to save memory
    unsigned num_nodes = nodes.size() / 3;
    std::vector<unsigned short, aligned_allocator<unsigned short> > nodes_is_leaf_feature_id(num_nodes);
    std::vector<float, aligned_allocator<float> > nodes_value(num_nodes);
    for (unsigned i = 0; i < num_nodes; ++i)
    {
        nodes_is_leaf_feature_id[i] = (unsigned short)nodes[i * 3] << 1;
        nodes_value[i] = nodes[i * 3 + 1];
        unsigned is_tree_leaf = nodes[i * 3 + 2];
        if (is_tree_leaf == 1)
        {
            nodes_is_leaf_feature_id[i] |= 0x1;
        }
    }

    std::cout << "Allocating hier format on FPGA" << std::endl;
    // Mod nodes layout

    cl_mem_ext_ptr_t prefix_sum_subtree_nums_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_prefix_sum_subtree_nums;

    for(int i = 0; i < NUM_SLRS; i++){
        prefix_sum_subtree_nums_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        prefix_sum_subtree_nums_alloc[i].param = 0;
        prefix_sum_subtree_nums_alloc[i].obj   = prefix_sum_subtree_nums.data();
        OCL_CHECK(err, cl::Buffer d_prefix_sum_subtree_nums_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * prefix_sum_subtree_nums.size(), &prefix_sum_subtree_nums_alloc[i], &err));
        d_prefix_sum_subtree_nums.push_back(d_prefix_sum_subtree_nums_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_prefix_sum_subtree_nums(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * prefix_sum_subtree_nums.size(), prefix_sum_subtree_nums.data(), &err));
    // Mod nodes layout
    cl_mem_ext_ptr_t nodes_value_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_nodes_value;

    for(int i = 0; i < NUM_SLRS; i++){
        nodes_value_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        nodes_value_alloc[i].param = 0;
        nodes_value_alloc[i].obj   = nodes_value.data();
        OCL_CHECK(err, cl::Buffer d_nodes_value_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(float) * nodes_value.size(), &nodes_value_alloc[i], &err));
        d_nodes_value.push_back(d_nodes_value_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_nodes_value(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float) * nodes_value.size(), nodes_value.data(), &err));
    cl_mem_ext_ptr_t nodes_is_leaf_feature_id_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_nodes_is_leaf_feature_id;

    for(int i = 0; i < NUM_SLRS; i++){
        nodes_is_leaf_feature_id_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        nodes_is_leaf_feature_id_alloc[i].param = 0;
        nodes_is_leaf_feature_id_alloc[i].obj   = nodes_is_leaf_feature_id.data();
        OCL_CHECK(err, cl::Buffer d_nodes_is_leaf_feature_id_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned short) * nodes_is_leaf_feature_id.size(), &nodes_is_leaf_feature_id_alloc[i], &err));
        d_nodes_is_leaf_feature_id.push_back(d_nodes_is_leaf_feature_id_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_nodes_is_leaf_feature_id(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned short) * nodes_is_leaf_feature_id.size(), nodes_is_leaf_feature_id.data(), &err));

    cl_mem_ext_ptr_t idx_to_subtree_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_idx_to_subtree;

    for(int i = 0; i < NUM_SLRS; i++){
        idx_to_subtree_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        idx_to_subtree_alloc[i].param = 0;
        idx_to_subtree_alloc[i].obj   = idx_to_subtree.data();
        OCL_CHECK(err, cl::Buffer d_idx_to_subtree_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * idx_to_subtree.size(), &idx_to_subtree_alloc[i], &err));
        d_idx_to_subtree.push_back(d_idx_to_subtree_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_idx_to_subtree(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * idx_to_subtree.size(), idx_to_subtree.data(), &err));
    cl_mem_ext_ptr_t leaf_idx_boundry_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_leaf_idx_boundry;

    for(int i = 0; i < NUM_SLRS; i++){
        leaf_idx_boundry_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        leaf_idx_boundry_alloc[i].param = 0;
        leaf_idx_boundry_alloc[i].obj   = leaf_idx_boundry.data();
        OCL_CHECK(err, cl::Buffer d_leaf_idx_boundry_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * leaf_idx_boundry.size(), &leaf_idx_boundry_alloc[i], &err));
        d_leaf_idx_boundry.push_back(d_leaf_idx_boundry_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_leaf_idx_boundry(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * leaf_idx_boundry.size(), leaf_idx_boundry.data(), &err));
    cl_mem_ext_ptr_t g_subtree_nodes_offset_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_g_subtree_nodes_offset;

    for(int i = 0; i < NUM_SLRS; i++){
        g_subtree_nodes_offset_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        g_subtree_nodes_offset_alloc[i].param = 0;
        g_subtree_nodes_offset_alloc[i].obj   = g_subtree_nodes_offset.data();
        OCL_CHECK(err, cl::Buffer d_g_subtree_nodes_offset_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * g_subtree_nodes_offset.size(), &g_subtree_nodes_offset_alloc[i], &err));
        d_g_subtree_nodes_offset.push_back(d_g_subtree_nodes_offset_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_g_subtree_nodes_offset(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * g_subtree_nodes_offset.size(), g_subtree_nodes_offset.data(), &err));
    cl_mem_ext_ptr_t g_subtree_idx_to_subtree_offset_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_g_subtree_idx_to_subtree_offset;

    for(int i = 0; i < NUM_SLRS; i++){
        g_subtree_idx_to_subtree_offset_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        g_subtree_idx_to_subtree_offset_alloc[i].param = 0;
        g_subtree_idx_to_subtree_offset_alloc[i].obj   = g_subtree_idx_to_subtree_offset.data();
        OCL_CHECK(err, cl::Buffer d_g_subtree_idx_to_subtree_offset_temp(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(unsigned) * g_subtree_idx_to_subtree_offset.size(), &g_subtree_idx_to_subtree_offset_alloc[i], &err));
        d_g_subtree_idx_to_subtree_offset.push_back(d_g_subtree_idx_to_subtree_offset_temp);
    }
    // OCL_CHECK(err, cl::Buffer d_g_subtree_idx_to_subtree_offset(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * g_subtree_idx_to_subtree_offset.size(), g_subtree_idx_to_subtree_offset.data(), &err));

    #ifdef BATCH
    // cl_mem_ext_ptr_t query_curr_subtree_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    // std::vector<cl::Buffer> d_query_curr_subtree;

    // for(int i = 0; i < NUM_SLRS; i++){
    //     query_curr_subtree_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
    //     query_curr_subtree_alloc[i].param = 0;
    //     query_curr_subtree_alloc[i].obj   = nullptr;
    //     OCL_CHECK(err, cl::Buffer d_query_curr_subtree_temp(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, sizeof(unsigned) * row, &query_curr_subtree_alloc[i], &err));
    //     d_query_curr_subtree.push_back(d_query_curr_subtree_temp);
    // }
    // // OCL_CHECK(err, cl::Buffer d_query_curr_subtree(context, CL_MEM_READ_WRITE, sizeof(unsigned) * row, nullptr, &err));
    // cl_mem_ext_ptr_t query_data_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    // std::vector<cl::Buffer> d_query_data;

    // for(int i = 0; i < NUM_SLRS; i++){
    //     query_data_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
    //     query_data_alloc[i].param = 0;
    //     query_data_alloc[i].obj   = nullptr;
    //     OCL_CHECK(err, cl::Buffer d_query_data_temp(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, sizeof(unsigned) * row, &query_data_alloc[i], &err));
    //     d_query_data.push_back(d_query_data_temp);
    // }
    // OCL_CHECK(err, cl::Buffer d_query_data(context, CL_MEM_READ_WRITE, sizeof(unsigned) * row, nullptr, &err));

    cl_mem_ext_ptr_t query_info_alloc[NUM_SLRS];  // Declaring extensions for multiple SLR buffers
    std::vector<cl::Buffer> d_query_info;

    for(int i = 0; i < NUM_SLRS; i++){
        unsigned num_queries;
        if(i < NUM_SLRS-1) num_queries = row/NUM_SLRS;
        else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
        query_info_alloc[i].flags = i | XCL_MEM_TOPOLOGY; // DDR[i]
        query_info_alloc[i].param = 0;
        query_info_alloc[i].obj   = nullptr;
        OCL_CHECK(err, cl::Buffer d_query_info_temp(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, sizeof(query_t) * num_queries * num_of_trees, &query_info_alloc[i], &err));
        d_query_info.push_back(d_query_info_temp);
    }
    #endif

    std::vector<cl::Event> write_event(NUM_SLRS);

    for(int i = 0; i < NUM_SLRS; i++){
        unsigned num_queries;
        if(i < NUM_SLRS-1) num_queries = row/NUM_SLRS;
        else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
        unsigned start = 0;
        int narg = 0;
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_of_trees));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_prefix_sum_subtree_nums[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_nodes_value[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_nodes_is_leaf_feature_id[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_idx_to_subtree[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_leaf_idx_boundry[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_g_subtree_nodes_offset[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_g_subtree_idx_to_subtree_offset[i]));
        #ifdef BATCH
        // OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_query_curr_subtree[i]));
        // OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_query_data[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_query_info[i]));
        #endif
        #if (NUM_CUS > 1)
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, start));
        #endif
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, num_queries));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, col));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_queries[i]));
        OCL_CHECK(err, err = rf_kernel[i*NUM_CUS].setArg(narg++, d_results[i]));

        std::cout << "Copying hier format to FPGA SLR " << i << std::endl;
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_prefix_sum_subtree_nums[i],
                                                        d_nodes_value[i],
                                                        d_nodes_is_leaf_feature_id[i],
                                                        d_idx_to_subtree[i],
                                                        d_leaf_idx_boundry[i],
                                                        d_g_subtree_nodes_offset[i],
                                                        d_g_subtree_idx_to_subtree_offset[i],
                                                        d_queries[i],
                                                        d_results[i]}, 0 /* 0 means from host*/,
                                                        nullptr, &write_event[i]));
    }

    #if SPLIT
    std::vector<cl::Event> rf_burst_event(NUM_SLRS);
    #endif

    std::vector<cl::Event> rf_event(NUM_SLRS*NUM_CUS);

    #if SPLIT
    for(int i = 0; i < NUM_SLRS; i++){
        unsigned num_queries;
        if(i < (NUM_SLRS-1)) num_queries = row/NUM_SLRS;
        else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
        unsigned start = 0;
        unsigned end = num_queries;
        std::cout << i << " Burst CU start is " << start << ", end is " << end << std::endl;
        int narg = 0;
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, num_of_trees));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_prefix_sum_subtree_nums[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_nodes_value[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_nodes_is_leaf_feature_id[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_idx_to_subtree[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_leaf_idx_boundry[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_g_subtree_nodes_offset[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_g_subtree_idx_to_subtree_offset[i]));
        #ifdef BATCH
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_query_info[i]));
        #endif
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, start));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, end));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, col));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_queries[i]));
        OCL_CHECK(err, err = rf_kernel_burst[i].setArg(narg++, d_results[i]));
    }
    #endif
    
    for(int i = 0; i < NUM_SLRS * NUM_CUS; i++){
        unsigned num_queries;
        if(i < (NUM_SLRS-1)*NUM_CUS) num_queries = row/NUM_SLRS;
        else num_queries = (row / NUM_SLRS) + (row % NUM_SLRS);
        unsigned num_queries_per_cu = num_queries / NUM_CUS;
        unsigned start = num_queries_per_cu * (i % NUM_CUS);
        unsigned end;
        if(i % NUM_CUS == NUM_CUS-1){
            end = num_queries;
        }
        else{
            end = num_queries_per_cu * ((i % NUM_CUS) + 1);
        }
        std::cout << i << " CU start is " << start << ", end is " << end << std::endl;
        int narg = 0;
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, num_of_trees));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_prefix_sum_subtree_nums[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_nodes_value[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_nodes_is_leaf_feature_id[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_idx_to_subtree[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_leaf_idx_boundry[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_g_subtree_nodes_offset[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_g_subtree_idx_to_subtree_offset[i / NUM_CUS]));
        #ifdef BATCH
        // OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_query_curr_subtree[i / NUM_CUS]));
        // OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_query_data[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_query_info[i / NUM_CUS]));
        #endif
        #if NUM_CUS > 1
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, start));
        #endif
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, end));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, col));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_queries[i / NUM_CUS]));
        OCL_CHECK(err, err = rf_kernel[i].setArg(narg++, d_results[i / NUM_CUS]));
    }
    START_TIMER
    #if SPLIT
    for(int i = 0; i < NUM_SLRS; i++){
        std::cout << "Start executing hier format burst on FPGA SLR" << i << std::endl;
        OCL_CHECK(err, err = q.enqueueTask(rf_kernel_burst[i], &write_event, &rf_burst_event[i]));
    }
    for(int i = 0; i < NUM_SLRS * NUM_CUS; i++){
        std::cout << "Start executing hier format on FPGA SLR" << i << std::endl;
        OCL_CHECK(err, err = q.enqueueTask(rf_kernel[i], &rf_burst_event, &rf_event[i]));
    }
    #else
    for(int i = 0; i < NUM_SLRS * NUM_CUS; i++){
        std::cout << "Start executing hier format on FPGA SLR" << i << std::endl;
        OCL_CHECK(err, err = q.enqueueTask(rf_kernel[i], &write_event, &rf_event[i]));
    }
    #endif

    
    // START_TIMER

    // narg = 0;
    // OCL_CHECK(err, err = generate_results_kernel.setArg(narg++, row));
    // OCL_CHECK(err, err = generate_results_kernel.setArg(narg++, num_of_trees));
    // OCL_CHECK(err, err = generate_results_kernel.setArg(narg++, d_results));
    // std::vector<cl::Event> gen_res_event(1);
    // OCL_CHECK(err, err = q.enqueueNDRangeKernel(generate_results_kernel, 0, 1, 1, &rf_event, &gen_res_event[0]));    

    std::vector<cl::Event> read_event(NUM_SLRS);

    for(int i = 0; i < NUM_SLRS; i++){
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_results[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &rf_event,
                                                        &read_event[i]));
    }
    
    OCL_CHECK(err, err = cl::Event::waitForEvents(read_event));
    STOP_TIMER("hier kernel")

    wrong_num = 0;
    for (size_t i = 0; i < row; ++i)
    {
        size_t results_array = i/(row/NUM_SLRS);
        if(results_array >= NUM_SLRS) results_array = NUM_SLRS - 1;
        size_t offset = results_array * (row/NUM_SLRS);
        // std::cout << "Accessing " << results_array << " " << i - offset << std::endl;
        if (h_results[results_array][i - offset] != y_test[i])
        {
            wrong_num++;
        }
    }
    std::cout << "hier result is wrong with this many: " << wrong_num << std::endl;
    std::cout << "accuracy rate: " << (float)(row - wrong_num) / (float)row << std::endl;
#endif
    // main returns
    return 0;
}

// predict the result over a decision_tree
float predict_tree_fpga_layout(int num_of_trees, const unsigned *prefix_sum_subtree_nums, const float *nodes, const unsigned *idx_to_subtree, const unsigned *leaf_idx_boundry, const unsigned *g_subtree_nodes_offset, const unsigned *g_subtree_idx_to_subtree_offset, unsigned tree_num, float *row)
{

    unsigned tree_off_set = prefix_sum_subtree_nums[tree_num];
    // unsigned num_of_subtrees = prefix_sum_subtree_nums[tree_num+1] - tree_off_set;

    unsigned curr_subtree_idx = 0;
    // iterate over subtree
    while (true)
    {
        // fetch the subtree nodes
        const float *subtree_node_list;
        subtree_node_list = nodes + g_subtree_nodes_offset[tree_off_set + curr_subtree_idx] * 3;

        // fetch subtree_leaf_idx_boundry
        const unsigned subtree_leaf_idx_boundry = leaf_idx_boundry[tree_off_set + curr_subtree_idx];

        // fetch subtree_idx_to_other_subtree
        const unsigned *subtree_idx_to_subtree = idx_to_subtree + g_subtree_idx_to_subtree_offset[tree_off_set + curr_subtree_idx] * 2;

        // start from node 0
        unsigned curr_node = 0;
        // iterate over nodes in a subtree
        while (true)
        {
            unsigned feature_id = subtree_node_list[curr_node * 3];
            float node_value = subtree_node_list[curr_node * 3 + 1];
            unsigned is_tree_leaf = subtree_node_list[curr_node * 3 + 2];

            // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
            if (is_tree_leaf == 1)
                return node_value;

            // if node is not leaf, we need two comparisons to decide if we keep traverse inside current subtree, or we go to another subtree
            bool not_subtree_bottom = curr_node < subtree_leaf_idx_boundry;
            bool go_left = row[feature_id] <= node_value;

            // if not reach bottom of subtree, keep iterating using 2*i+1 or 2*i+2
            if (not_subtree_bottom)
            {
                // go to left child in subtree
                if (go_left)
                    curr_node = curr_node * 2 + 1;
                // go to right child in subtree
                else
                    curr_node = curr_node * 2 + 2;
                // if reach bottom of subtree, then we need to go to another subtree
            }
            else
            {
                unsigned leaf_idx = curr_node - subtree_leaf_idx_boundry;
                if (go_left)
                    curr_subtree_idx = subtree_idx_to_subtree[2 * leaf_idx];
                else
                    curr_subtree_idx = subtree_idx_to_subtree[2 * leaf_idx + 1];
                // stop the iterating of the current subtree, jump to the outer loop
                break;
            }
        }
    }
}

template <typename T>
unsigned read_arr(std::ifstream &infile, std::vector<T, aligned_allocator<T>> &output, std::string var_name)
{
    std::string str;
    char c;
    infile >> str;
    if (str != var_name)
    {
        std::cout << str << "error reading " << var_name << std::endl;
    }
    unsigned len;
    infile >> len >> c;
    output.resize(len);
    for (unsigned i = 0; i < len; ++i)
    {
        infile >> output[i] >> c;
    }
    //  std::cout << "Read " << str << " with " << len << " elements\n";
    return len;
}

template <typename T>
void read_2darr(std::ifstream &infile, std::vector<T, aligned_allocator<T>> &output, std::string var_name, unsigned &row, unsigned &col)
{
    std::string str;
    char c;
    infile >> str;
    if (str != var_name)
    {
        std::cout << str << "error reading " << var_name << std::endl;
    }
    unsigned nrow, ncol;
    infile >> nrow >> c >> ncol >> c;
    row = nrow;
    col = ncol;
    output.resize(nrow * ncol);
    for (unsigned i = 0; i < nrow * ncol; ++i)
    {
        infile >> output[i] >> c;
    }
    //  std::cout << "Read " << str << " with " << nrow << " rows" << " and " << ncol << " cols.\n";
}

//    node_list = tree[1]
//    edge_list = tree[2]
//    node_is_leaf = tree[3]
//    node_features = tree[4]
//    node_values = tree[5]
float predict_tree_csr_layout(unsigned *node_list, unsigned *edge_list, unsigned *node_is_leaf, unsigned *node_features, float *node_values, float *row)
{
    // start from node 0
    unsigned curr_node = 0;
    // iterate over nodes in a subtree
    while (true)
    {
        unsigned feature_id = node_features[curr_node];
        float node_value = node_values[curr_node];
        unsigned is_tree_leaf = node_is_leaf[curr_node];
        // if node is leaf, then the prediction is over, we return the predicted value in node_value (in a tree leaf, node_value holds the predicted result)
        if (is_tree_leaf == 1)
            return node_value;
        // if node is not leaf, we need two comparisons to decide if we keep traverse
        bool go_left = row[feature_id] <= node_value;
        if (go_left)
            curr_node = edge_list[node_list[curr_node]];
        // go to right child in subtree
        else
            curr_node = edge_list[node_list[curr_node] + 1];
    }
}