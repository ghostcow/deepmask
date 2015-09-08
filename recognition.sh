#!/bin/bash
RESULT_DIR=$1
MATLAB_PATH="/usr/local/MATLAB/R2013a/bin/matlab"

cd recognition
$MATLAB_PATH -nodesktop -nojvm -nosplash -r "resultDir='$RESULT_DIR';main_jointBayesian;exit"
