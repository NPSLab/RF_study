#!/bin/bash

cp ./test_input.txt ./tree_input.txt

for sd in 10 
do
  echo "TTT$sd: test tree depth $sd"
  cp ./td${sd}_csr.txt ./treefile_csr.txt
  ./rf_gpu_csr
  echo
  #echo "TTT$sd: test subtree depth 3"
  #cp ./td${sd}sd2_hier.txt ./treefile_hier.txt
  #./rf_gpu
  #echo
  #echo "TTT$sd: test subtree depth 4"
  #cp ./td${sd}sd3_hier.txt ./treefile_hier.txt
  #./rf_gpu
  #echo
  echo "TTT$sd: test subtree depth 5 with default layout"
  cp ./td${sd}sd4_hier.txt ./treefile_hier.txt
  ./rf_gpu_iter
  echo "TTT$sd: test subtree depth 5 with new layout 1"
  ./rf_gpu_iter_new_layout
  echo "TTT$sd: test subtree depth 5 with new layout 2"
  ./rf_gpu_iter_new_layout2
done
