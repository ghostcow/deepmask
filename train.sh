#!/usr/bin/env bash

cd nn

# Train
th doall.lua --gpu 1 \
             --splitName smallval2014 \
             --dataPath /home/lioruzan/obj_detection_proj/mscoco.torch/annotations \
             --imageDirPath /home/lioruzan/obj_detection_proj/data/coco/images \
             --epochs 10 \
             --learningRate 0.001 \
             --batchSize 1 \
             --epochSize 2000 \
             --nWorkers 0