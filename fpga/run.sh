#!/bin/bash
FILEDIR=$1
KERNEL=$2
KERNELNAME=$3
SUBDEPTH=$4
BATCH=$5
NUM_CUS=$6
NUM_SLRS=$7
MODE=$8
SPLIT=$9
EXTENSION=

if [ "$MODE" = "FPGA_CSR" ]; then
    EXTENSION=/*csr.txt
else
    EXTENSION=/*hier.txt
fi

for fullname in $FILEDIR$EXTENSION
do
    filename=$(basename -- "$fullname")
    for ((i=0; i<5; i++))
    do
        make run TARGET=hw PLATFORM=xilinx_u250_xdma_201830_2 HOST=x86 KERNEL=$KERNEL SUBDEPTH=$SUBDEPTH TREEFILE=$FILEDIR$filename INPUTFILE="${FILEDIR}"test_input.txt NUM_SLRS=1 BATCH=$BATCH NUM_CUS=$NUM_CUS NUM_SLRS=$NUM_SLRS MODE=$MODE SPLIT=$SPLIT >> ~/RF_study/data/super_test/fpga_output/"${filename%.*}".${KERNELNAME}_$i.FPGA.log
        mv ${KERNEL}*.link.xclbin.run_summary  ~/RF_study/data/super_test/fpga_output/"${filename%.*}".${KERNELNAME}_$i.xclbin.run_summary
    done
done
