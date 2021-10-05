#!/bin/bash

cp ./test_input.txt ./tree_input.txt

for tree_depth in 15 16 
do
  echo
  echo "TTT$tree_depth: test tree depth $tree_depth"
  cp ./td${tree_depth}_csr.txt ./treefile_csr.txt
  ./gpu/rf_gpu_csr ${tree_depth}

  echo
  echo "TTT$tree_depth: test subtree depth 5"
  cp ./td${tree_depth}sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_iter ${tree_depth}
done
