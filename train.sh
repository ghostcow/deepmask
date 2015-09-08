#!/usr/bin/env bash

cd nn
LAYER=13

# Train
th doall.lua --dataPath  ~/results/scratch/teacher_new_init/results/dataset.t7 \
             --retrain ~/results/scratch/reduce_and_reuse/power_2_75_FT/layer_${LAYER}/coef2_reduced_model.t7 \
             --save ~/results/scratch/reduce_and_reuse/power_2_75_FT/layer_${LAYER}/results_FT \
             --momentum 0.9 \
             --learningRate 0.0001 \
             --batchSize 64 \
             --epochs 10
