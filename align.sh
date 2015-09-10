#!/bin/bash
cd 2D_Alignment

MATLAB_PATH="/usr/local/MATLAB/R2013a/bin/matlab"

# alignment script parameters
MAIN_DIR="/home/adampolyak/datasets/CASIA"
IMAGES_DIR="${MAIN_DIR}/images"
DETECT_FILE="detections.txt"
# scratch generation script parameters
DEEPID_DIR="$MAIN_DIR/aligned_deepid"

ALIGN_CMD="addpath('/media/data/projects/mexopencv/');mainDir='$MAIN_DIR';allImagesDir='$IMAGES_DIR';detectionsFileName='$DETECT_FILE';AlignDataset;exit"

$MATLAB_PATH -nodesktop -nojvm -nosplash -r $ALIGN_CMD
