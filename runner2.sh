#!/bin/bash

source "/home/ivo/Env/keras/bin/activate"
export PYTHONPATH=$PYTHONPATH:"/home/ivo/Code/VAE/dapper/mods/ComplexCircle"
MODELPATH="dapper/mods/ComplexCircle/"

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
