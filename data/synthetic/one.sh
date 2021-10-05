#!/bin/bash

cp ./test_input.txt ./tree_input.txt

for sd in 15 
do
  echo
  echo "TTT$sd: test tree depth $sd"
  cp ./td${sd}_csr.txt ./treefile_csr.txt
  ./gpu/rf_gpu_csr

  echo
  echo "TTT$sd: test subtree depth 5"
  cp ./td${sd}sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_iter
done
