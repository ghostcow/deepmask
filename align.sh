#!/bin/bash
cd 2D_Alignment

MATLAB_PATH="/usr/local/MATLAB/R2013a/bin/matlab"

# alignment script parameters
MAIN_DIR="/media/data/datasets/FaCeleb"
IMAGES_DIR="/media/data/datasets/FaCeleb/images"
DETECT_FILE="detections.txt"
# scratch generation script parameters
DEEPID_DIR="$MAIN_DIR/aligned_deepid"
SCRATCH_DIR="$MAIN_DIR/aligned_scratch"

ALIGN_CMD="addpath('/media/data/projects/mexopencv/');mainDir='$MAIN_DIR';allImagesDir='$IMAGES_DIR';detectionsFileName='$DETECT_FILE';AlignDataset;exit"
GENERATE_SCRATCH_CMD="mainDir='$MAIN_DIR';srcDir='$DEEPID_DIR';dstDir='$SCRATCH_DIR';FLIP=false;GenerateScratchFormat;exit"

$MATLAB_PATH -nodesktop -nojvm -nosplash -r $ALIGN_CMD
$MATLAB_PATH -nodesktop -nojvm -nosplash -r $GENERATE_SCRATCH_CMD
