#!/bin/bash
# myArray=("covtype_fixed" "susy_fixed" "higgs_fixed" "covtype_bigst" "susy_bigst" "higgs_bigst")
myArray=("covtype_bigst" "susy_bigst" "higgs_bigst")

make clean
for dataset in ${myArray[@]}; do
  ./run.sh ~/RF_study/data/super_test/${dataset}/ fpga_random_forest_csr csr 0 0 1 1 FPGA_CSR 0
  ./run.sh ~/RF_study/data/super_test/${dataset}/ fpga_random_forest_naive_reorder_buffer_res_cu_rep best_naive 0 0 12 4 FPGA_HIER 0
  ./run.sh ~/RF_study/data/super_test/${dataset}/ fpga_random_forest_hybrid_cu_rep_split hybrid_split 12 1 10 4 FPGA_HIER 1
done