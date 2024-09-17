#!/bin/bash

source "/home/ivo/Env/keras/bin/activate"
export PYTHONPATH=$PYTHONPATH:"/home/ivo/Code/VAE/dapper/mods/ComplexCircle"
MODELPATH="dapper/mods/ComplexCircle/"
FILENAME=$1
MINRANGE=$2
MAXRANGE=$3
STEP=(100 7)

for ((minseeds=$MINRANGE;minseeds<$MAXRANGE;minseeds=$minseeds+${STEP[0]}));
do
    maxseeds=$(($minseeds+${STEP[0]}))
    if [ $maxseeds -gt $MAXRANGE ]; then maxseeds=$MAXRANGE; fi
    for ((minseed=$minseeds;$minseed<$maxseeds;minseed=$minseed+${STEP[1]}));
    do
	maxseed=$(($minseed+${STEP[1]}))
	if [ $maxseed -gt $maxseeds ]; then maxseed=$maxseeds; fi
	echo "RUNNING SEEDS "$minseed $maxseed
	python ${MODELPATH}${FILENAME} $minseed $maxseed
    done
done
