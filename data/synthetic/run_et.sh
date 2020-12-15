#!/bin/bash
cp ./test_input.txt ./tree_input.txt

for sd in 7 8 9 10 11 12 13 14 15
do
  echo
  echo
  echo
  echo
  echo "TTT$sd: test tree depth $sd"
  echo "TTT$sd: test subtree depth 3"
  cp ./td${sd}sd2_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_et3
  echo
  echo "TTT$sd: test subtree depth 4"
  cp ./td${sd}sd3_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_et4
  echo
  echo "TTT$sd: test subtree depth 5"
  cp ./td${sd}sd4_hier.txt ./treefile_hier.txt
  ./gpu/rf_gpu_et5
done
