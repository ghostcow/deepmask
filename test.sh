#!/usr/bin/env bash

cd nn
LAYER=3

# Test
th doall.lua --dataPath ~/results/scratch/teacher_new_init/results/dataset.t7 \
             --retrain ~/results/scratch/reduce_and_reuse/PCA_75/layer_${LAYER}/pca_model.nat \
             --batchSize 128 \
             --save ~/results/scratch/reduce_and_reuse/PCA_75/layer_${LAYER}/test_results \
             --nWorkers 0 \
             --testOnly