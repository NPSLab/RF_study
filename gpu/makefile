all: gpu_csr gpu_hier_iter gpu_hier_et3 gpu_hier_et4 gpu_hier_et5 

gpu_csr: gpu_random_forest.cu
	nvcc -std=c++11 -DGPU_CSR  -O3 -Wno-deprecated-gpu-targets gpu_random_forest.cu -o rf_gpu_csr

gpu_hier_iter: gpu_random_forest.cu
	nvcc -std=c++11 -DGPU_HIER -DITER -O3 -Wno-deprecated-gpu-targets gpu_random_forest.cu -o rf_gpu_iter

gpu_hier_et3: gpu_random_forest.cu
	nvcc -std=c++11 -DGPU_HIER -DET3  -O3 -Wno-deprecated-gpu-targets gpu_random_forest.cu -o rf_gpu_et3

gpu_hier_et4: gpu_random_forest.cu
	nvcc -std=c++11 -DGPU_HIER -DET4  -O3 -Wno-deprecated-gpu-targets gpu_random_forest.cu -o rf_gpu_et4

gpu_hier_et5: gpu_random_forest.cu
	nvcc -std=c++11 -DGPU_HIER -DET5  -O3 -Wno-deprecated-gpu-targets gpu_random_forest.cu -o rf_gpu_et5

clean:
	rm rf_gpu_*

