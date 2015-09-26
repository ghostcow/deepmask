#!/bin/bash

RESULT_DIR=$1
DATASET=~/results/teacher_new_init/results/dataset.t7

cd recognition
th extract_features.lua --modelPath $RESULT_DIR/nn/model_best.net \
			            --outputPath $RESULT_DIR/features \
			            --dataPath $DATASET
th extract_features_lfw.lua  --modelPath $RESULT_DIR/nn/model_best.net \
			                 --outputPath $RESULT_DIR/features
