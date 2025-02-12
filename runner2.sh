#!/bin/bash

source "~/Env/keras/bin/activate"
export PYTHONPATH=$PYTHONPATH:"~/Code/VAE/dapper/mods/ComplexCircle"
export MODELPATH="dapper/mods/ComplexCircle/"
export CUDA_VISIBLE_DEVICES=""

FILENAME=$1
MAX_INDEX=$2
MIN_EXP=$3
MAX_EXP=$4

exit_code=0
for ((exp=$MIN_EXP;$exp<=$MAX_EXP;exp=$exp+1));
do
    for ((index=0;$index<=$MAX_INDEX;index=$index+1));
    do
        echo "RUNNING EXPERIMENT " $index $exp
        exit_code=$(python ${MODELPATH}${FILENAME} $index $exp)
        echo "EXIT " $exit_code
    done
 
    python ${MODELPATH}${FILENAME} -1 $exp
done    
