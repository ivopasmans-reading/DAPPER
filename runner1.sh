#!/bin/bash

source "/home/ivo/Env/keras/bin/activate"
export PYTHONPATH=$PYTHONPATH:"/home/ivo/Code/VAE/dapper/mods/ComplexCircle"
MODELPATH="dapper/mods/ComplexCircle/"

FILENAME=$1
MAX_INDEX=$2

exit_code=0
for ((index=0;$index<=$MAX_INDEX;index=$index+1));
do
    echo "RUNNING EXPERIMENT " $index
    exit_code=$(python ${MODELPATH}${FILENAME} $index)
    echo "EXIT " $exit_code
done

python ${MODELPATH}${FILENAME} -1
    
