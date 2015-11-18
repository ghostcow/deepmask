#!/usr/bin/env bash

cd nn

# Train
th doall.lua --gpu 0 \
             --splitName smallval2014 \
             --dataPath /home/lioruzan/obj_detection_proj/mscoco.torch/annotations \
             --imageDirPath /home/lioruzan/obj_detection_proj/data/coco/images \
             --epochs 1 \
             --learningRate 0.001 \
             --batchSize 32 \
             --epochSize 1