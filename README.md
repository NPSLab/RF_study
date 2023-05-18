# Accelerating Random Forest Classification on GPU and FPGA

A repository for accelerating random forest inference on GPU and FPGA. ([Paper](https://doi-org/10.1145/3545008.3545067))

### Dependencies
Generating Forests:
  - Follow installation instructions at https://rapids.ai/start.html#get-rapids to prepare your environment for CUML
  - Version 21.08 of the RAPIDS API Conda environment were used for this

GPU:
  - CUDA 11.6 was used
FPGA:
  - Vitis 2020.2 was used

### /gpu directory  - contains different GPU kernels
- To generate all gpu random forest traverers, type "make all" inside gpu folder
- Eight versions will be generated
  - rf_gpu_csr: CSR traversal 
  - rf_gpu_iter: Naive traversal
  - rf_gpu_et3: Encoded Topology Traversal (SD=3)
  - rf_gpu_et4: Encoded Topology Traversal (SD=4)
  - rf_gpu_et5: Encoded Topology Traversal (SD=5)
  - rf_gpu_rootsubtree: Rootsubtree Variant
  - rf_gpu_iter_new_layout: Variant of Naive Traversal with Different Node Encoding
  - rf_gpu_iter_new_layout2: Variant of Naive Traversal with Different Node Encoding
- rf_gpu_csr uses CSR format gpu inputs
- All others use hierarchical format gpu inputs
- Other kernel variant codes are present, add to the Makefile or run NVCC compilation manually to generate binaries

### /data directory- contains scripts to generate forests and run CUML baseline
- Make sure to download the appropriate dataset from the UCI Machine Learning repository (https://archive.ics.uci.edu/ml/index.php) and place data into appropriate dataset folder
- Run rf_fixed.py to generate forests with fixed maximum subtree depth.
  - NE_RANGE, TD_RANGE, SD_RANGE configure parameters for number of trees, maximum tree depths, and maximum subtree depths, respectively
  - The forest is stored in a file called "clustered_fixed_DATASET_tdX_neY_sdZ_hier.txt" (hierarchical format) or "clustered_fixed_DATASET_tdX_neY_csr.txt" (CSR format)
    - DATASET = name of the dataset
    - X = maximum tree depth
    - Y = number of trees
    - Z = maximum subtree depth
- Run rf_bigst.py to generate forests with a specified root subtree maximum depth, configured with the INIT_RANGE variable
  - The forest is stored in a file called "clustered_var_DATASET_tdX_neY_sdZ_istdW_hier.txt" (hierarchical format) or "clustered_fixed_DATASET_tdX_neY_csr.txt" (CSR format)
    - DATASET = name of the dataset
    - X = maximum tree depth
    - Y = number of trees
    - Z = maximum subtree depth
    - W = root subtree maximum depth


### Running the GPU traversal
1. To run a test on GPU, you need one of the binaries generated in the /gpu directory.
2. Have two files in the same folder.
  - For rf_gpu_csr, you need:
    - inputfile - tree_input.txt
    - treefile  - treefile_csr.txt

  - For hierarchical gpu codes, you need:
    - inputfile - tree_input.txt
    - treefile  - treefile_hier.txt
3. tree_input.txt is the same as test_input.txt created from running Python forest generation scripts
4. treefile inputs are generated from the same forest generation script as step 3. Rename one of the forest files to treefile_csr.txt for CSR or treefile_hier.txt for hierarchical traversal kernels

### Compiling and running the FPGA traversa1. 
- Use run.sh to automate both the hardware and host compilation and execution of the random forest traversal.
- An example can be seen in run_tests.sh - this example generates the results shown in Fig. 9

_For manual use of the makefile, arguments are described below:_
### fpga/Makefile Arguments
- **TARGET**: Determine hardware, simulation, or emulation (hw, hw_emu, sw_emu)
- **HOST_ARCH**: x86
- **KERNEL**: Name of .cpp kernel file located in /fpga
- **SUBDEPTH**: Max subdepth for on-chip memory storage in hybrid version. Scales by 2^SUBDEPTH. Set to 0 for CSR/independent version, >0 for hier version.
- **CONFIG**: Name of the .cfg file used for hardware generation
- **TREEFILE**: The forest file, either ending with \_csr.txt or \_hier.txt, depending on CSR or HIER version
- **INPUTFILE**: Input file (same as step 3. of GPU traversal, same as test_input.txt from generation)
- **BATCH**: Whether the kernel performs batching (0, 1)
- **NUM_CUS**: Number of Compute Units (1 -> 12) _Make sure .cfg file matches this number_
- **NUM_SLRS**: Number of Super Logic Regions (1, 4) _Make sure .cfg file matches this number_
- **MODE**: Baseline CSR version (FPGA_CSR) or independent/hybrid versions (FPGA_HIER)
- **KERNEL_NAME**: hier_kernel
