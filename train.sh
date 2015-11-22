#!/usr/bin/env bash

cd nn

# calculating by negative ratio = 0.5, where half the sgd updates are mask and half score-
# at 428002 instances in the train set that fit my selection criteria,
# at 32 images per batch- 17833 batches per epoch
# at 64 samples per batch- 8916 batches per epoch

# Train
th doall.lua --gpu 1 \
             --splitName train2014 \
             --dataPath /home/lioruzan/obj_detection_proj/mscoco.torch/annotations \
             --imageDirPath /home/lioruzan/obj_detection_proj/data/coco/images \
             --epochs 100 \
             --learningRate 0.001 \
             --batchSize 64 \
             --epochSize 8916 \
             --nWorkers 4