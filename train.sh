#!/usr/bin/env bash

cd nn

# Original command
#th doall.lua --imageDirPath  /home/adampolyak/datasets/CASIA/aligned_scratch \
#             --imageSize 1x100x100 \
#             --split 10 \
#             --netType scratch \
#             --save ~/results/scratch_blur \
#             --blurSize 5 \
#             --blurSigma 10 \
#             --momentum 0.9 \
#             --learningRate 0.001 \
#             --batchSize 128 \
#             --epochs 10

# Train
th doall.lua --dataPath  /home/adampolyak/results/scratch_blur/dataset.t7    \
             --netType scratch \
             --save ~/results/scratch_blur/results_1 \
             --blurSize 5 \
             --blurSigma 10 \
             --momentum 0.9 \
             --learningRate 0.01 \
             --batchSize 128