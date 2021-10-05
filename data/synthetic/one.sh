#!/bin/bash

cp ./test_input.txt ./tree_input.txt
# declare -a depths = ( "20" "40" "45" "45" "45" "60" )
# declare -a estimators = ( "50" "50" "10" "50" "100" "50" )

# for ne in
for tree_depth in 20 40 45 60
do
  # tree_depth = 20
  # ne = 50
  echo
  echo "TTT$tree_depth: test tree depth $tree_depth"
  cp ./input_files/clusteredcovtype_td${tree_depth}_ne50_csr.txt ./treefile_csr.txt
  ./gpu/rf_gpu_csr ${tree_depth}

  echo
  echo "TTT$tree_depth: test subtree depth 5"
  cp ./input_files/clusteredcovtype_td${tree_depth}_ne50_sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_iter ${tree_depth}
done

for ne in 10 50 100
do
  # tree_depth = 20
  # ne = 50
  echo
  echo "TTT45: test tree depth 45"
  cp ./input_files/clusteredcovtype_td45_ne${ne}_csr.txt ./treefile_csr.txt
  ./gpu/rf_gpu_csr 45

  echo
  echo "TTT45: test subtree depth 5"
  cp ./input_files/clusteredcovtype_td45_ne${ne}_sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_iter 45
done