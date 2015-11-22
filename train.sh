#!/usr/bin/env bash

cd nn

# calculating by negative ratio = 0.5, where half the sgd updates are mask and half score-
# at 428002 instances in the train set that fit my selection criteria,
# at 32 images per batch- 17833 batches per epoch
# at 64 samples per batch- 8916 batches per epoch

# Train
# luajit -joff doall.lua --gpu 1 \

th doall.lua --gpu 1 \
             --splitName train2014 \
             --dataPath /home/lioruzan/obj_detection_proj/mscoco.torch/annotations \
             --imageDirPath /home/lioruzan/obj_detection_proj/data/coco/images \
             --epochs 100 \
             --learningRate 0.001 \
             --batchSize 32 \
             --epochSize 17833 \
             --nWorkers 4

# For testing
#th doall.lua --gpu 1 \
#             --splitName train2014 \
#             --dataPath /home/lioruzan/obj_detection_proj/mscoco.torch/annotations \
#             --imageDirPath /home/lioruzan/obj_detection_proj/data/coco/images \
#             --epochs 100 \
#             --learningRate 0.001 \
#             --batchSize 1 \
#             --epochSize 1 \
#             --nWorkers 0
