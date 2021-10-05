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
  cp ./input_files/susy_format/clusteredSUSY_td${tree_depth}_ne50_csr.txt ./treefile_csr.txt
  ./gpu/rf_gpu_csr ${tree_depth}

  echo
  echo "TTT$tree_depth: test subtree depth 5"
  cp ./input_files/susy_format/clusteredSUSY_td${tree_depth}_ne50_sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_iter ${tree_depth}
done
