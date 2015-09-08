#!/bin/bash

RESULT_DIR=~/results/fitnet_1/KD/results_4
DATASET=~/results/teacher_new_init/results/dataset.t7

cd recognition
th extract_features.lua --modelPath $RESULT_DIR/nn/model.net \
			            --outputPath $RESULT_DIR/features \
			            --dataPath $DATASET
th extract_features_lfw.lua  --modelPath $RESULT_DIR/nn/model.net \
			                 --outputPath $RESULT_DIR/features
